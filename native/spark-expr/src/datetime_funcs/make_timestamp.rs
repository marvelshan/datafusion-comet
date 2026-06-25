// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, Decimal128Array, Int32Array, StringArray, TimestampMicrosecondArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, TimeUnit};
use chrono::{NaiveDate, NaiveDateTime, TimeZone};
use chrono_tz::Tz;
use datafusion::common::DataFusionError;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMakeTimestamp {
    signature: Signature,
    fail_on_error: bool,
}

impl SparkMakeTimestamp {
    pub fn new(fail_on_error: bool) -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(7)],
                Volatility::Immutable,
            ),
            fail_on_error,
        }
    }
}

impl Default for SparkMakeTimestamp {
    fn default() -> Self {
        Self::new(false)
    }
}
fn invalid_timestamp_message(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
) -> String {
    if !(1..=12).contains(&month) {
        return format!("Invalid value for MonthOfYear (valid values 1 - 12): {month}");
    }
    if !(0..=23).contains(&hour) {
        return format!("Invalid value for HourOfDay (valid values 0 - 23): {hour}");
    }
    if !(0..=59).contains(&minute) {
        return format!("Invalid value for MinuteOfHour (valid values 0 - 59): {minute}");
    }
    if !(0..=59).contains(&second) {
        return format!("Invalid value for SecondOfMinute (valid values 0 - 59): {second}");
    }
    if !(1..=31).contains(&day) {
        return format!("Invalid value for DayOfMonth (valid values 1 - 28/31): {day}");
    }
    if day == 29 && month == 2 {
        return format!("Invalid date 'February 29' as '{year}' is not a leap year");
    }
    if NaiveDate::from_ymd_opt(year, month as u32, day as u32).is_none() {
        return format!("Invalid date '{year}-{month:02}-{day:02}'");
    }
    format!("Invalid date '{year}-{month:02}-{day:02}'")
}

fn make_timestamp_utc(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    sec: i32,
    micros: i64,
    tz: &Tz,
) -> Option<i64> {
    let date = NaiveDate::from_ymd_opt(year, month as u32, day as u32)?;
    let time = chrono::NaiveTime::from_hms_micro_opt(hour as u32, minute as u32, sec as u32, micros as u32)?;
    let local_dt = NaiveDateTime::new(date, time);
    let utc_dt = tz.from_local_datetime(&local_dt).single()?;
    Some(utc_dt.timestamp_micros())
}

fn cast_to_int32(arr: &Arc<dyn Array>) -> Result<Arc<dyn Array>, DataFusionError> {
    if arr.data_type() == &DataType::Int32 {
        Ok(Arc::clone(arr))
    } else {
        cast(arr.as_ref(), &DataType::Int32)
            .map_err(|e| DataFusionError::Execution(format!("Failed to cast to Int32: {e}")))
    }
}

impl ScalarUDFImpl for SparkMakeTimestamp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "make_timestamp"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType, DataFusionError> {
        Ok(DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue, DataFusionError> {
        if args.args.len() != 7 {
            return Err(DataFusionError::Execution(format!(
                "make_timestamp requires 7 arguments, got {}",
                args.args.len()
            )));
        }

        let year = &args.args[0];
        let month = &args.args[1];
        let day = &args.args[2];
        let hour = &args.args[3];
        let minute = &args.args[4];
        let second = &args.args[5];
        let timezone_arg = &args.args[6];

        let num_rows = args
            .args
            .iter()
            .find_map(|arg| match arg {
                ColumnarValue::Array(array) => Some(array.len()),
                ColumnarValue::Scalar(_) => None,
            })
            .unwrap_or(1);

        let tz_str = match timezone_arg {
            ColumnarValue::Scalar(s) => match s {
                datafusion::scalar::ScalarValue::Utf8(Some(s)) | datafusion::scalar::ScalarValue::Utf8View(Some(s)) => s.clone(),
                _ => "UTC".to_string(),
            },
            ColumnarValue::Array(arr) => {
                if arr.len() > 0 {
                    if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
                        sa.value(0).to_string()
                    } else {
                        "UTC".to_string()
                    }
                } else {
                    "UTC".to_string()
                }
            }
        };

        let tz: Tz = tz_str.parse().unwrap_or(chrono_tz::UTC);

        let year_arr = cast_to_int32(&year.clone().into_array(num_rows)?)?;
        let month_arr = cast_to_int32(&month.clone().into_array(num_rows)?)?;
        let day_arr = cast_to_int32(&day.clone().into_array(num_rows)?)?;
        let hour_arr = cast_to_int32(&hour.clone().into_array(num_rows)?)?;
        let minute_arr = cast_to_int32(&minute.clone().into_array(num_rows)?)?;
        let second_arr = second.clone().into_array(num_rows)?;

        let year_array = year_arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            DataFusionError::Execution("make_timestamp: failed to cast year to Int32".to_string())
        })?;
        let month_array = month_arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            DataFusionError::Execution("make_timestamp: failed to cast month to Int32".to_string())
        })?;
        let day_array = day_arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            DataFusionError::Execution("make_timestamp: failed to cast day to Int32".to_string())
        })?;
        let hour_array = hour_arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            DataFusionError::Execution("make_timestamp: failed to cast hour to Int32".to_string())
        })?;
        let minute_array = minute_arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            DataFusionError::Execution("make_timestamp: failed to cast minute to Int32".to_string())
        })?;
        let second_array = second_arr
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "make_timestamp: expected Decimal128 for second argument".to_string(),
                )
            })?;

        const MICROS_PER_SECOND: i128 = 1_000_000;

        let len = year_array.len();
        let mut builder = TimestampMicrosecondArray::builder(len).with_timezone("UTC");

        for i in 0..len {
            if year_array.is_null(i)
                || month_array.is_null(i)
                || day_array.is_null(i)
                || hour_array.is_null(i)
                || minute_array.is_null(i)
                || second_array.is_null(i)
            {
                builder.append_null();
            } else {
                let y = year_array.value(i);
                let mo = month_array.value(i);
                let d = day_array.value(i);
                let h = hour_array.value(i);
                let mi = minute_array.value(i);
                let sec_unscaled = second_array.value(i);

                let full_secs = sec_unscaled.div_euclid(MICROS_PER_SECOND);
                let frac_micros = sec_unscaled.rem_euclid(MICROS_PER_SECOND);

                if full_secs < 0 || full_secs > 59 {
                    if self.fail_on_error {
                        return Err(DataFusionError::Execution(format!(
                            "Invalid value for SecondOfMinute (valid values 0 - 59): {full_secs}"
                        )));
                    }
                    builder.append_null();
                    continue;
                }

                let secs = full_secs as i32;
                let micros = frac_micros as i64;

                match make_timestamp_utc(y, mo, d, h, mi, secs, micros, &tz) {
                    Some(micros_since_epoch) => builder.append_value(micros_since_epoch),
                    None => {
                        if self.fail_on_error {
                            return Err(DataFusionError::Execution(invalid_timestamp_message(
                                y, mo, d, h, mi, secs,
                            )));
                        }
                        builder.append_null();
                    }
                }
            }
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_timestamp_valid() {
        let tz: Tz = "UTC".parse().unwrap();
        assert!(make_timestamp_utc(2024, 6, 15, 10, 30, 45, 123456, &tz).is_some());
        assert!(make_timestamp_utc(1970, 1, 1, 0, 0, 0, 0, &tz).is_some());
        assert!(make_timestamp_utc(2024, 12, 31, 23, 59, 59, 999999, &tz).is_some());
    }

    #[test]
    fn test_make_timestamp_epoch() {
        let tz: Tz = "UTC".parse().unwrap();
        let epoch_micros = make_timestamp_utc(1970, 1, 1, 0, 0, 0, 0, &tz).unwrap();
        assert_eq!(epoch_micros, 0);
    }

    #[test]
    fn test_make_timestamp_invalid_month() {
        let tz: Tz = "UTC".parse().unwrap();
        assert_eq!(make_timestamp_utc(2024, 0, 1, 0, 0, 0, 0, &tz), None);
        assert_eq!(make_timestamp_utc(2024, 13, 1, 0, 0, 0, 0, &tz), None);
    }

    #[test]
    fn test_make_timestamp_invalid_day() {
        let tz: Tz = "UTC".parse().unwrap();
        assert_eq!(make_timestamp_utc(2024, 6, 0, 0, 0, 0, 0, &tz), None);
        assert_eq!(make_timestamp_utc(2024, 6, 32, 0, 0, 0, 0, &tz), None);
        assert_eq!(make_timestamp_utc(2024, 2, 30, 0, 0, 0, 0, &tz), None);
    }

    #[test]
    fn test_make_timestamp_invalid_hour() {
        let tz: Tz = "UTC".parse().unwrap();
        assert_eq!(make_timestamp_utc(2024, 6, 15, 25, 0, 0, 0, &tz), None);
        assert_eq!(make_timestamp_utc(2024, 6, 15, -1, 0, 0, 0, &tz), None);
    }
}
