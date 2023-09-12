// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2022 Vladimir Pinchuk (https://github.com/VladimirP1)

use crate::gyro_source::GyroSource;
use itertools::izip;
use nalgebra::{ComplexField, Vector3};
use rand::Rng;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::iter::zip;
use log::info;


use std::fs::File;
use std::io::Result;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::path::Path;
use telemetry_parser::{Input, util::{self, IMUData}};
use std::collections::BTreeMap;
use telemetry_parser::tags_impl::{GetWithType, GroupId, TagId, TimeQuaternion};
pub type TimeIMU = telemetry_parser::util::IMUData;
use nalgebra::*;
pub type Quat64 = UnitQuaternion<f64>;
pub type TimeQuat = BTreeMap<i64, Quat64>; // key is timestamp_us
pub struct FileMetadata {
    pub imu_orientation: Option<String>,
    pub raw_imu:  Option<Vec<TimeIMU>>,
    pub quaternions:  Option<TimeQuat>,
    pub detected_source: Option<String>
}
pub struct OptimSync {
    sample_rate: f64,
    gyro: [Vec<f64>; 3],
    raw_gyro: GyroSource,
}

fn blackman(width: usize) -> Vec<f32> {
    let a0 = 7938.0 / 18608.0;
    let a1 = 9240.0 / 18608.0;
    let a2 = 1430.0 / 18608.0;
    let mut samples = vec![0.0; width];
    let size = (width - 1) as f32;
    for i in 0..width {
        let n = i as f32;
        let v = a0 - a1 * (2.0 * PI * n / size).cos() + a2 * (4.0 * PI * n / size).cos();
        samples[i] = v;
    }
    samples
}

impl OptimSync {
    pub fn new(gyro: &GyroSource) -> Option<OptimSync> {
        let duration_ms = gyro.raw_imu.last()?.timestamp_ms - gyro.raw_imu.first()?.timestamp_ms;
        let samples_total = gyro.raw_imu.iter().filter(|x| x.gyro.is_some()).count();
        let avg_sr = samples_total as f64 / duration_ms * 1000.0;

        let interp_gyro = |ts| {
            let i_r = gyro
                .raw_imu
                .partition_point(|sample| sample.timestamp_ms < ts)
                .min(gyro.raw_imu.len() - 1);
            let i_l = i_r.max(1) - 1;

            let left = &gyro.raw_imu[i_l];
            let right = &gyro.raw_imu[i_r];
            if i_l == i_r {
                return Vector3::from_column_slice(&left.gyro.unwrap_or_default());
            }
            (Vector3::from_column_slice(&left.gyro.unwrap_or_default()) * (right.timestamp_ms - ts)
                + Vector3::from_column_slice(&right.gyro.unwrap_or_default()) * (ts - left.timestamp_ms))
                / (right.timestamp_ms - left.timestamp_ms)
        };

        let mut gyr = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
        for i in 0..((duration_ms * avg_sr / 1000.0) as usize) {
            let s = interp_gyro(i as f64 * 1000.0 / avg_sr);
            for j in 0..3 {
                gyr[j].push(s[j]);
            }
        }

        Some(OptimSync {
            sample_rate: avg_sr,
            gyro: gyr,
            raw_gyro: gyro.clone(),
        })
    }

    pub fn run2(
        &mut self,
        target_sync_points: usize,
        trim_start_s: f64,
        trim_end_s: f64,
    ) -> Vec<f64> {
        let gyro_c32: Vec<Vec<Complex<f32>>> = self
            .gyro
            .iter()
            .map(|v| v.iter().map(|&x| Complex::from_real(x as f32)).collect())
            .collect();

        let step_size_samples = 16;
        let nms_radius = ((self.sample_rate / 16.0 / 2.0) * 8.0) as usize; // sync points no closer than 8 seconds

        let fft_size = self.sample_rate.round() as usize;
        let scale = (1.0 / fft_size as f32).sqrt() / fft_size as f32 * 256.0;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);

        let win = blackman(fft_size);

        let ffts: Vec<Vec<_>> = gyro_c32
            .iter()
            .map(|gyro_c32_chan| {
                gyro_c32_chan
                    .windows(fft_size)
                    .step_by(step_size_samples)
                    .map(|chunk| {
                        let mut cm: Vec<_> = zip(chunk, &win).map(|(x, y)| x * y).collect();
                        fft.process(&mut cm);
                        zip(cm.iter(), cm.iter().rev())
                            .take(fft_size / 2)
                            .map(|(a, b)| a + b)
                            .map(|x| x.norm() * scale)
                            .collect::<Vec<_>>()
                    })
                    .collect()
            })
            .collect();

        let map_to_bin = |freq: f64| {
            (fft_size as f64 / self.sample_rate * freq)
                .round()
                .max(0.0)
                .min((fft_size / 2 - 1) as f64) as usize
        };

        let band_energy = |axis: &Vec<Vec<f32>>, begin, end| {
            let f: Vec<_> = axis
                .iter()
                .map(|bins| bins[map_to_bin(begin)..map_to_bin(end)].iter().sum::<f32>())
                .collect();
            f
        };

        fn sum_vec_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            zip(a, b).map(|(a, b)| a + b).collect()
        }
        let merged_ffts: Vec<_> = izip!(&ffts[0], &ffts[1], &ffts[2])
            .map(|(x, y, z)| sum_vec_f32(&sum_vec_f32(x, y), z))
            .collect();

        let lf = band_energy(&merged_ffts, 0.0, 2.0);
        let mf = band_energy(&merged_ffts, 2.0, 30.0);
        let hf = band_energy(&merged_ffts, 30.0, 2000.0);

        let mut rank: Vec<_> = izip!(&lf, &mf, &hf)
            .map(|(lf, mf, hf)| {
                // we do not like low freqs and high freqs, but mid freqs are good
                mf / (1.0 + nlfunc(*hf, 450.0) * 0.003) / (1.0 + nlfunc(*lf, 650.0) * 0.003)
            })
            .collect();

        for i in 0..rank.len() {
            if rank[i] < 100.0
                || (i * step_size_samples) as f64 / self.sample_rate < trim_start_s
                || (i * step_size_samples) as f64 / self.sample_rate > trim_end_s
            {
                rank[i] = 0.0;
            }
        }

        let mut rank_nms = rank.clone();
        for i in 0..rank.len() {
            for j in
                (i as i64 - nms_radius as i64).max(0) as usize..(i + nms_radius).min(rank.len() - 1)
            {
                if rank[j] < rank[i] {
                    rank_nms[j] = 0.0;
                }
            }
        }

        let mut sync_points = Vec::<f64>::new();
        for i in 0..rank.len() {
            if rank_nms[i] > 0.1 {
                sync_points.push(
                    (i as f64 * step_size_samples as f64 + fft_size as f64 / 2.0)
                        / self.sample_rate
                        * 1000.0,
                );
            }
        }

        let mut selected_sync_points = Vec::<f64>::new();
        let mut rng = rand::thread_rng();
        for _ in 0..target_sync_points {
            if sync_points.is_empty() { break; }
            let rnd = rng.gen_range(trim_start_s * 1000.0..trim_end_s * 1000.0);
            let mut p = sync_points.partition_point(|x| x < &rnd).min(sync_points.len() - 1);
            if (sync_points[(p as i64-1).max(0) as usize] - rnd).abs() < (sync_points[p] - rnd) {
                p -= 1;
            }
            selected_sync_points.push(sync_points[p]);
            sync_points.remove(p);
        }

        // use inline_python::python;
        // python! {
        //     import matplotlib.pyplot as plt
        //     import os

        //     plt.plot('lf, label = "lf", alpha = .3)
        //     plt.plot('mf, label = "mf", alpha = .3)
        //     plt.plot('hf, label = "hf", alpha = .3)

        //     plt.plot('rank, label = "rank")
        //     plt.plot('rank_nms, label = "rank_nms")

        //     plt.legend()
        //     plt.tight_layout()
        //     fig = plt.gcf()
        //     fig.set_size_inches(10, 5)
        //     plt.show()
        // }
        selected_sync_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        selected_sync_points
    }

pub fn parse_telemetry_file(path: &str) -> Result<FileMetadata> {
	let progress_cb = |p| {
            let now = std::time::Instant::now();
        };
	let mut stream = File::open(path)?;
	let filesize = stream.metadata()?.len() as usize;

	let filename = Path::new(&path).file_name().unwrap().to_str().unwrap();

	let input = Input::from_stream(&mut stream, filesize, filename, progress_cb, Arc::new(AtomicBool::new(false)))?;

	let mut detected_source = input.camera_type();
	if let Some(m) = input.camera_model() { detected_source.push(' '); detected_source.push_str(m); }

	let mut imu_orientation = None;
	let mut quaternions = None;

	println!("file: {}", filename);
	println!("filesize: {}", filesize);
	println!("detected_source: {}", detected_source);

	// Get IMU orientation and quaternions
	if let Some(ref samples) = input.samples {
		let mut quats = TimeQuat::new();
		for info in samples {
			if let Some(ref tag_map) = info.tag_map {
				if let Some(map) = tag_map.get(&GroupId::Quaternion) {
					if let Some(arr) = map.get_t(TagId::Data) as Option<&Vec<TimeQuaternion<f64>>> {
						for v in arr {
							quats.insert((v.t * 1000.0) as i64, Quat64::from_quaternion(Quaternion::from_parts(
								v.v.w, 
								Vector3::new(v.v.x, v.v.y, v.v.z)
							)));
						}
					}
				}
				if let Some(map) = tag_map.get(&GroupId::Gyroscope) {
					let mut io = match map.get_t(TagId::Orientation) as Option<&String> {
						Some(v) => v.clone(),
						None => "XYZ".into()
					};
					io = input.normalize_imu_orientation(io);
					imu_orientation = Some(io);
				}
			}
		}
		if !quats.is_empty() {
			quaternions = Some(quats);
		}
	}

	let raw_imu = util::normalized_imu(&input, None).ok();

	Ok(FileMetadata {
            imu_orientation,
            detected_source: Some(detected_source),
            quaternions,
            raw_imu
        })
	

// let x = [0u32, 1, 2];
// let y = [3u32, 4, 5];
// let mut fg = Figure::new();
// fg.axes2d()
// .lines(&x, &y, &[Caption("A line"), Color("black")]);
// fg.show();

}


pub fn run(
    &mut self,
    target_sync_points: usize,
    trim_start_s: f64,
    trim_end_s: f64,
    still_threshold: f64, 
    flippy_threshold: f64
) -> Vec<f64>{
	let sync_duration_s = 1.5;
    let mut num_sync_points = 3;
    let total = get_total_angular_velocity(&self.raw_gyro.raw_imu);
	let len = total.len();
	let timestamps_ms = get_timestamps_ms(&self.raw_gyro.raw_imu);
	let window =  get_sample_rate(&timestamps_ms) as usize;
	let total_mean = rolling(&total, window, 0);
	let total_median = rolling(&total, window, 1);
	let total_std = rolling(&total, window, 2);
	let total_gradient = gradient(&total, &timestamps_ms);
	// let total_gradient_mean = rolling(&total_gradient, window, 0);
	let mut rating:Vec<f64> = Vec::new();
	for i in 0..len{
		rating.push((1.0 + 2.0 * total_std[i].abs()) / (1.0 + total_mean[i]));
	}

	// thresholding on rating
	let sync_padding: usize = ((sync_duration_s * window as f64) * 1.2 / 2.0) as usize;
	for i in 0..len -1{
		if total_mean[i] > flippy_threshold || total_median[i] < still_threshold || total_gradient[i] > 200.0{
			for jj in max(0, i as i16 - sync_padding as i16) as usize..min(len - 1, i + sync_padding){
				rating[jj] = 0.0;
			}
		}
	}
	let(start_points, end_points) = get_movement(&total_median, 0.02);
	let (mut start_index, mut end_index) = get_main_movement(&start_points, &end_points, total.len());

	// use the whole data if movement shorter than 5 seconds
	if timestamps_ms[end_index] - timestamps_ms[start_index] < 5000.0{
		start_index = 0;
		end_index = len - 1;
	}

	num_sync_points = (((timestamps_ms[end_index] - timestamps_ms[start_index]) / (sync_duration_s * 1000.0 * 1.5)) as usize).min(num_sync_points).max(1);
	let sync_part_duration = (end_index - start_index) / num_sync_points;
	let mut sync_point_timestamp_ms: Vec<f64> = Vec::new();
	for i in 0..num_sync_points{
		let start = start_index + i * sync_part_duration + sync_padding;
		let end = start_index + (i + 1) * sync_part_duration - sync_padding;
		let mut max_idx = start;
		let mut max = 0.0;
		for jj in 0..(end - start -1){
			if rating[jj + start] > max{
				max = rating[jj + start];
				max_idx = jj + start;
			}
		}
		sync_point_timestamp_ms.push(timestamps_ms[max_idx]);
	}

    info!("Suggested Syncpoints: {:?}", sync_point_timestamp_ms);
	// for debugging, I don't like rust plotting yet
	// use inline_python::python;
	// python!{
	// 	import matplotlib.pyplot as plt
	// 	import os
	// 	if os.path.isfile("mjr_dark.mplstyle"):
	// 		plt.style.use("mjr_dark.mplstyle")
	// 	total = 'total
	// 	ts = 'timestamps_ms[:len(total)]
	// 	plt.plot(ts, 'total, label="total")
	// 	plt.plot(ts, 'total_std, label="total_std")
	// 	plt.plot(ts, 'rating, label="rating")
	// 	for val in 'sync_point_timestamp_ms:
	// 		plt.axvspan(val - 'sync_duration_s / 2 * 1000, val + 'sync_duration_s / 2 * 1000, facecolor='g', alpha=0.3, label="syncpoint")
	// 	plt.legend()
	// 	plt.tight_layout()
	// 	fig = plt.gcf()
	// 	fig.set_size_inches(10, 5)
	// 	plt.show()
	// }
	// sync_point_timestamp_ms
	// vec![1.0]
	sync_point_timestamp_ms
}
}


fn mean(data: &[f64]) -> Option<f64> {
    let sum = data.iter().sum::<f64>() as f64;
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f64),
        _ => None,
    }
}

fn std_deviation(data: &[f64]) -> Option<f64> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data.iter().map(|value| {
                let diff = data_mean - (*value as f64);

                diff * diff
            }).sum::<f64>() / count as f64;

            Some(variance.sqrt())
        },
        _ => None
    }
}

fn gradient(f: &Vec<f64>, timestamps_ms: &Vec<f64>) -> Vec<f64>{
	// gradient by central difference, edges by for and backward difference
	let mut gradient: Vec<f64> = Vec::new();
	let len = f.len();
	assert!(len >= 2);
	gradient.push((f[1] - f[0])/(timestamps_ms[1] - timestamps_ms[0])*1000.0);
	for i in 1..len - 1{
		gradient.push((f[i+1] - f[i-1])/(timestamps_ms[i+1] - timestamps_ms[i-1])*1000.0)
	}
	gradient.push((f[len-1] - f[len-2])/(timestamps_ms[len-1] - timestamps_ms[len-2])*1000.0);
	gradient
}

pub fn median(orig_data: &[f64]) -> f64 {
    let length = orig_data.len();

    if length == 0 {
        return f64::NAN;
    }
	let mut data = orig_data.to_vec();
	// println!("{}", length);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

	// println!("{:#?}", data);

    if length % 2 == 0 {
        let left = data[length / 2 - 1];
        let right = data[length / 2];
        (left as f64 + right as f64) / 2.0
    } else {
		data[length / 2] as f64
    }
}

// Creates a type alias
fn rolling(data: &Vec<f64>, mut window: usize, mode: usize) -> Vec<f64> {
	let length: usize = data.len();
	let mut roll: Vec<f64> = Vec::new();
	// make window size uneven
	if window % 2 == 0{
		window += 1;
	}
	assert!(window < length);
	let edge: usize = window / 2;

	for (ii, _item) in data.iter().enumerate(){
		if ii < edge || ii > (length - edge - 1) {
			// TODO edge implementation
			roll.push(0.0);
		}else {
			// println!("{:?}", &data[ii - edge..ii + edge + 1]);
			if mode == 0 {
				roll.push(mean(&data[ii - edge..ii + edge + 1]).unwrap());
			}
			if mode == 1 {
				roll.push(median(&data[ii - edge..ii + edge + 1]));
			}
			if mode == 2 {
				roll.push(std_deviation(&data[ii - edge..ii + edge + 1]).unwrap());
			}
		}
	}
	roll
}

fn get_sample_rate(timestamps_ms: &Vec<f64>) -> f64{
	timestamps_ms.len() as f64 / (timestamps_ms[timestamps_ms.len()-1] - timestamps_ms[0]) * 1000.0
}

fn get_total_angular_velocity(im: &Vec<IMUData>) -> Vec<f64> {
	let mut timestamp: Vec<f64> = Vec::new();
	let mut total: Vec<f64> = Vec::new();
	// let mut timestamp: Vec<double> = Vec::new();
	for item in im {
		timestamp.push(item.timestamp_ms);
		let gyro = item.gyro;
		match gyro {
			Some(gyro) => total.push((gyro[0].powi(2) + gyro[1].powi(2) + gyro[2].powi(2)).sqrt() / 180.0 * std::f64::consts::PI),
			_ => (),
		};
	}
	total
}

fn get_timestamps_ms(im: &Vec<IMUData>) -> Vec<f64> {
	let mut timestamp: Vec<f64> = Vec::new();
	for i in im {
		timestamp.push(i.timestamp_ms);
	}
	timestamp
}

fn get_movement(total_angular_velocity_median: &Vec<f64>, threshold: f64) -> (Vec<usize>, Vec<usize>){
	let mut start_points: Vec<usize> = Vec::new();
	let mut end_points: Vec<usize> = Vec::new();
	// make sure start point is always before end point
	let mut previous_moving: bool = false;
	for (ii, item) in total_angular_velocity_median.iter().enumerate(){
		if item > &threshold && !previous_moving{
			start_points.push(ii);
			previous_moving = true;
		}else if item < &threshold && previous_moving{
			end_points.push(ii);
			previous_moving = false;
		}
	}
	// make sure there is always an end point to a start point
	if end_points.len() < start_points.len(){
		end_points.push(total_angular_velocity_median.len() - 1);
	}
	(start_points, end_points)
}

fn get_main_movement(start_points: &Vec<usize>, end_points: &Vec<usize>, len: usize) -> (usize, usize){
	let mut weighted_duration: Vec<usize> = Vec::new();
	for ii in 0..start_points.len(){
		// assuming uniform logging rate
		let duration: usize = end_points[ii] - start_points[ii];
		// add linear weight with max in the middle of the video
		let mid: usize = start_points[ii] + duration/2;
		let mut factor: f32 = mid as f32 / len as f32;
		if mid > len/2 {
			factor = 1.0 - factor;
		}
		weighted_duration.push((duration as f32 * factor) as usize)
	}
	// get longest rated duration
	let main_movement = weighted_duration.iter().enumerate().fold((0, 0), |max, (ind, &val)| if val > max.1 {(ind, val)} else {max});
	(start_points[main_movement.0], end_points[main_movement.0])
}

pub fn nlfunc(arg: f32, trip_point: f32) -> f32 {
    if arg < trip_point {
        0.0
    } else {
        arg - trip_point
    }
}
