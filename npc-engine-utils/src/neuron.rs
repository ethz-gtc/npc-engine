use std::iter::zip;

use rand::Rng;

const RELU_LEAK: f32 = 0.01;

#[derive(Debug, Clone)]
/// A simple leaky ReLU neuron with I inputs.
pub struct Neuron<const I: usize> {
	pub weights: [f32; I],
	pub bias: f32,
}
impl<const I: usize> Neuron<I> {
	/// Creates a new neuron with weights and bias to 0.
	pub fn zero() -> Self {
		Self {
			weights: [0.; I],
			bias: 0.
		}
	}
	/// Creates a new neuron with random weights within [-1, 1] and bias to 0.
	pub fn random_with_0_bias() -> Self {
		let mut rng = rand::thread_rng();
		Self {
			weights: [0.; I].map(|_| rng.gen_range(-1.0..=1.0)),
			bias: 0.0
		}
	}
	/// Creates a new neuron with random weights and bias within [-1, 1].
	pub fn random() -> Self {
		let mut rng = rand::thread_rng();
		Self {
			weights: [0.; I].map(|_| rng.gen_range(-1.0..=1.0)),
			bias: rng.gen_range(-1.0..=1.0)
		}
	}
	/// Creates a new neuron with random weights and bias within [-range, range].
	pub fn random_with_range(range: f32) -> Self {
		let mut rng = rand::thread_rng();
		Self {
			weights: [0.; I].map(|_| rng.gen_range(-range..=range)),
			bias: rng.gen_range(-range..=range)
		}
	}
	fn leaky_relu(x: f32) -> f32 {
		if x > 0. {
			x
		} else {
			RELU_LEAK * x
		}
	}
	fn leaky_relu_derivative(x: f32) -> f32 {
		if x > 0. {
			1.
		} else {
			RELU_LEAK
		}
	}
	fn weighted_sum(&self, x: &[f32; I]) -> f32 {
		zip(x, &self.weights).map(|(x, w)| x * w).sum::<f32>() + self.bias
	}
	/// Computes the neuron's output for x.
	pub fn output(&self, x: &[f32; I]) -> f32 {
		Self::leaky_relu(self.weighted_sum(x))
	}
	fn derivative(&self, x: &[f32; I]) -> f32 {
		Self::leaky_relu_derivative(self.weighted_sum(x))
	}
	fn compute_update(&self, x: &[f32; I], value: f32, e: f32) -> ([f32; I], f32) {
		let d_y_e_der = -self.derivative(x) * value * e;
		(
			x.map(|x_i| d_y_e_der * x_i),
			d_y_e_der
		)
	}
	fn update_d_weights(d_weights_j: &[f32; I], d_bias_j: f32, d_weights: &mut [f32; I], d_bias: &mut f32) {
		for (d_weight, d_weight_j) in zip(d_weights.iter_mut(), d_weights_j) {
			*d_weight += d_weight_j;
		}
		*d_bias += d_bias_j;
	}
	fn update_d_weights_output(&self, x: &[f32; I], y_data: f32, e: f32, d_weights: &mut [f32; I], d_bias: &mut f32) {
		let y_pred = self.output(x);
		let (d_weights_j, d_bias_j) = self.compute_update(x, y_pred - y_data, e);
		Self::update_d_weights(&d_weights_j, d_bias_j, d_weights, d_bias);
	}
	fn update_d_weights_hidden(&self, x: &[f32; I], d_hidden: f32, e: f32, d_weights: &mut [f32; I], d_bias: &mut f32) {
		let (d_weights_j, d_bias_j) = self.compute_update(x, d_hidden, e);
		Self::update_d_weights(&d_weights_j, d_bias_j, d_weights, d_bias);
	}
	fn update_weights(&mut self, d_weights: &[f32; I], d_bias: f32) {
		for (w, d_w) in self.weights.iter_mut().zip(d_weights) {
			*w += d_w;
		}
		self.bias += d_bias;
	}
	/// Trains the neuron from data using back-propagation with epsilon learning rate (per data entry).
	pub fn train<'a>(&mut self, data: impl Iterator<Item=&'a ([f32; I], f32)>, epsilon: f32) {
		assert!(epsilon < 0.5);
		let e = 2. * epsilon;
		let mut d_weights = [0f32; I];
		let mut d_bias = 0f32;
		for (x, y_data) in data {
			self.update_d_weights_output(x, *y_data, e, &mut d_weights, &mut d_bias);
		}
		self.update_weights(&d_weights, d_bias);
	}
}

/// A simple neural network with I inputs and one hidden layer of H neurons.
#[derive(Debug, Clone)]
pub struct NetworkWithHiddenLayer<const I: usize, const H: usize> {
	pub hidden_layer: [Neuron<I>; H],
	pub output_layer: Neuron<H>
}
impl<const I: usize, const H: usize> NetworkWithHiddenLayer<I, H> {
	fn x_mid(&self, x: &[f32; I]) -> [f32; H] {
		self.hidden_layer
			.iter()
			.map(
				|n| n.output(x)
			)
			.collect::<Vec<_>>()
			.try_into()
			.unwrap()
	}
	/// Computes the network's output for x.
	pub fn output(&self, x: &[f32; I]) -> f32 {
		self.output_layer.output(&self.x_mid(x))
	}
	/// Trains the network from data using back-propagation with epsilon learning rate (per data entry).
	pub fn train<'a>(&mut self, data: impl Iterator<Item=&'a ([f32; I], f32)>, epsilon: f32) {
		assert!(epsilon < 0.5);
		let e = 2. * epsilon;
		// initialize updates to 0
		let mut d_weights_hidden = [[0f32; I]; H];
		let mut d_bias_hidden = [0f32; H];
		let mut d_weights_output = [0f32; H];
		let mut d_bias_output = 0f32;
		// process all training samples, queuing updates
		for (x, y_data) in data {
			// forward phase, intermediate layer
			let x_mid = self.x_mid(x);
			// back-propagation phase, output layer
			self.output_layer.update_d_weights_output(&x_mid, *y_data, e, &mut d_weights_output, &mut d_bias_output);
			// back-propagation phase, hidden layer
			let y_pred = self.output_layer.output(&x_mid);
			let d_output = self.output_layer.derivative(&x_mid) * (y_pred - y_data);
			for ((neuron, w_output), (d_weights, d_bias)) in zip(
				zip(self.hidden_layer.iter_mut(), self.output_layer.weights),
				zip(d_weights_hidden.iter_mut(), d_bias_hidden.iter_mut())
			) {
				let d_hidden = d_output * w_output;
				neuron.update_d_weights_hidden(x, d_hidden, e, d_weights, d_bias);
			}
		}
		// apply deltas
		for (neuron, (d_weights, d_bias)) in zip(
			self.hidden_layer.iter_mut(), zip(d_weights_hidden.iter(), d_bias_hidden)
		) {
			neuron.update_weights(d_weights, d_bias);
		}
		self.output_layer.update_weights(&d_weights_output, d_bias_output);
	}
}


#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::NetworkWithHiddenLayer;

    use super::Neuron;

	fn approx_equal(a: f32, b: f32) -> bool {
		(a - b).abs() < 1e-4
	}
	fn assert_approx_equal(a: f32, b: f32) {
		if !approx_equal(a, b) {
			panic!("{a} is different than {b}");
		}
	}

	const LINEAR_1D_DATA: [([f32; 1], f32); 3] = [
		([0.], 1.),
		([1.], 2.5),
		([2.], 4.),
	];

	#[test]
	fn linear_function_1d() {
		let mut neuron = Neuron::zero();
		for _i in 0..100 {
			neuron.train(LINEAR_1D_DATA.iter(), 0.1);
		}
		assert_approx_equal(neuron.weights[0], 1.5);
		assert_approx_equal(neuron.bias, 1.);
	}

	#[test]
	fn linear_function_2d() {
		let mut neuron = Neuron {
			weights: [0.234, -1.43],
			bias: -1.425
		};
		let data = [
			([0., 0.], 1.),
			([1., 0.], 1.5),
			([0., 1.], 2.),
			([1., 1.], 2.5),
		];
		for _i in 0..150 {
			neuron.train(data.iter(), 0.1);
		}
		assert_approx_equal(neuron.weights[0], 0.5);
		assert_approx_equal(neuron.weights[1], 1.0);
		assert_approx_equal(neuron.bias, 1.);
	}

	#[test]
	fn two_layers_optimal_must_be_stable() {
		let mut network = NetworkWithHiddenLayer {
			hidden_layer: [
				Neuron {
					weights: [1.0],
					bias: 0.0
				},
			],
			output_layer: Neuron {
				weights: [1.5],
				bias: 1.0
			}
		};
		for _ in 0..100 {
			network.train(LINEAR_1D_DATA.iter(), 0.1);
		}
		assert_approx_equal(network.hidden_layer[0].weights[0], 1.0);
		assert_approx_equal(network.hidden_layer[0].bias, 0.0);
		assert_approx_equal(network.output_layer.weights[0], 1.5);
		assert_approx_equal(network.output_layer.bias, 1.0);
		network = NetworkWithHiddenLayer {
			hidden_layer: [
				Neuron {
					weights: [1.5],
					bias: 1.0
				},
			],
			output_layer: Neuron {
				weights: [1.0],
				bias: 0.0
			}
		};
		for _ in 0..100 {
			network.train(LINEAR_1D_DATA.iter(), 0.1);
		}
		assert_approx_equal(network.hidden_layer[0].weights[0], 1.5);
		assert_approx_equal(network.hidden_layer[0].bias, 1.0);
		assert_approx_equal(network.output_layer.weights[0], 1.0);
		assert_approx_equal(network.output_layer.bias, 0.0);
	}

	#[test]
	fn linear_function_1d_hidden() {
		let mut min_sse = f32::INFINITY;
		for _rerun in 0..20 {
			let mut network = NetworkWithHiddenLayer {
				hidden_layer: [
					Neuron::<1>::random(),
				],
				output_layer: Neuron::random()
			};
			for _i in 0..500 {
				network.train(LINEAR_1D_DATA.iter(), 0.05);
			}
			let sse: f32 = LINEAR_1D_DATA.iter()
				.map(|([x], y)| {
					let y_pred = network.output(&[*x]);
					(y - y_pred) * (y - y_pred)
				})
				.sum();
			min_sse = sse.min(min_sse);
			// println!("{_rerun}: {sse}");
			// for ([x], y) in LINEAR_1D_DATA {
			// 	println!("{x}: {} (expected {y})", network.output(&[x]));
			// }
			// println!("{network:?}");
		}
		// The best min SSE over 20 runs must be close to 0
		assert!(min_sse < 0.01, "min SSE {min_sse} is >= 0.01 over 20 runs");
	}

	#[test]
	fn non_linear_function_1d() {
		// 2nd degree non-linear function, we are interested on range -2..2
		fn f(x: f32) -> f32 {
			x*x
		}
		let mut network = NetworkWithHiddenLayer {
			hidden_layer: [
				Neuron::<1>::random(),
				Neuron::random(),
				Neuron::random(),
				Neuron::random(),
				Neuron::random(),
				Neuron::random(),
				Neuron::random(),
				Neuron::random(),
			],
			output_layer: Neuron::random()
		};
		let mut rng = rand::thread_rng();
		let test_set = (0..100).map(|_| {
			let x = rng.gen_range(-2.0..2.0);
			(x, f(x))
		}).collect::<Vec<_>>();
		let count = 1000;
		let mut min_avg_sse = f32::INFINITY;
		for _rerun in 0..20 {
			for batch in 0..count {
				// generate new data and train
				let data = (0..4).map(|_| {
					let x = rng.gen_range(-2.0..2.0);
					([x], f(x))
				}).collect::<Vec<_>>();
				let progress = batch as f32 / count as f32;
				let epsilon = 0.02 * (1.0 - progress) + 0.01;
				network.train(data.iter(), epsilon);
				// test on the test set
				// let sse: f32 = test_set.iter()
				// 	.map(|(x, y)| {
				// 		let y_pred = network.output(&[*x]);
				// 		(y - y_pred) * (y - y_pred)
				// 	})
				// 	.sum();
				// if batch % 10 == 0 {
				// 	println!("{batch}: {sse}");
				// }
			}
			let sse: f32 = test_set.iter()
				.map(|(x, y)| {
					let y_pred = network.output(&[*x]);
					(y - y_pred) * (y - y_pred)
				})
				.sum();
			min_avg_sse = (sse  / test_set.len() as f32).min(min_avg_sse);
			// for x_i in -20..20 {
			// 	let x = x_i as f32 * 0.1;
			// 	let y = network.output(&[x]);
			// 	println!("{x}, {y}");
			// }
		}
		// The best min SSE over 20 runs must be close to 0
		assert!(min_avg_sse < 0.1, "min average SSE {min_avg_sse} is >= 0.1 over 20 runs");
	}

	#[test]
	fn xor() {
		let xor_data = [
			([0., 0.], 0.),
			([1., 0.], 1.),
			([0., 1.], 1.),
			([1., 1.], 0.)
		];
		let mut min_sse = f32::INFINITY;
		for _rerun in 0..20 {
			let mut network = NetworkWithHiddenLayer {
				hidden_layer: [
					Neuron::<2>::random_with_0_bias(),
					Neuron::random_with_0_bias(),
				],
				output_layer: Neuron::random_with_0_bias()
			};
			for _i in 0..1000 {
				network.train(xor_data.iter(), 0.03);
			}
			let mut sse = 0.;
			for (x, y) in xor_data {
				let y_pred = network.output(&x);
				sse += (y_pred - y) * (y_pred - y);
			}
			min_sse = sse.min(min_sse);
			// println!("{:?}", network);
			// for (x, y) in xor_data {
			// 	let y_pred = network.output(&x);
			// 	println!("{x:?}: {y_pred} (should be: {y})");
			// }
		}
		// The best min SSE over 20 runs must be close to 0
		assert!(min_sse < 0.01, "min SSE {min_sse} is >= 0.01 over 20 runs");
	}
}