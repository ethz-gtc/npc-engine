use std::iter::zip;

const RELU_LEAK: f32 = 0.01;

/// A simple leaky ReLU neuron
pub struct Neuron<const I: usize> {
	pub weights: [f32; I],
	pub bias: f32,
}
impl<const I: usize> Neuron<I> {
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
	pub fn output(&self, x: &[f32; I]) -> f32 {
		Self::leaky_relu(self.weighted_sum(x))
	}
	pub fn train<'a>(&mut self, data: impl Iterator<Item=&'a ([f32; I], f32)>, epsilon: f32) {
		assert!(epsilon < 0.5);
		let e = 2. * epsilon;
		let mut d_weights = [0f32; I];
		let mut d_bias = 0f32;
		for (x, y_data) in data {
			let y_pred = self.output(x);
			let w_sum = self.weighted_sum(x);
			let der = Self::leaky_relu_derivative(w_sum);
			let d_y_e_der = e * (y_data - y_pred) * der;
			for (i, x_i) in x.iter().enumerate() {
				d_weights[i] += d_y_e_der * x_i;
			}
			d_bias += d_y_e_der;
		}
		for (w, d_w) in self.weights.iter_mut().zip(d_weights) {
			*w += d_w;
		}
		self.bias += d_bias;
	}
}


#[cfg(test)]
mod tests {
    use super::Neuron;

	fn approx_equal(a: f32, b: f32) -> bool {
		(a - b).abs() < 1e-4
	}

	#[test]
	fn linear_function_1d() {
		let mut neuron = Neuron {
			weights: [0.],
			bias: 0.
		};
		let data = [
			([0.], 1.),
			([1.], 2.5),
			([2.], 4.),
		];
		for _i in 0..100 {
			neuron.train(data.iter(), 0.1);
		}
		assert!(approx_equal(neuron.weights[0], 1.5));
		assert!(approx_equal(neuron.bias, 1.));
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
		assert!(approx_equal(neuron.weights[0], 0.5));
		assert!(approx_equal(neuron.weights[1], 1.0));
		assert!(approx_equal(neuron.bias, 1.));
	}
}