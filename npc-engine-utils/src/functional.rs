/// Returns a reference to the first element of a tuple reference.
pub fn keep_first<A, B>(tuple: &(A, B)) -> &A {
	&tuple.0
}

/// Returns a reference to the second element of a tuple reference.
pub fn keep_second<A, B>(tuple: &(A, B)) -> &B {
	&tuple.1
}

/// Returns a mutable reference to the first element of a mutable tuple reference.
pub fn keep_first_mut<A, B>(tuple: &mut (A, B)) -> &mut A {
	&mut tuple.0
}

/// Returns a mutable reference to the second element of a mutable tuple reference.
pub fn keep_second_mut<A, B>(tuple: &mut (A, B)) -> &mut B {
	&mut tuple.1
}