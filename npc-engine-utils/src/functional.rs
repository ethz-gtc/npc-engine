pub fn keep_first<'a, A, B>(tuple: &'a (A, B)) -> &'a A {
	&tuple.0
}

pub fn keep_second<'a, A, B>(tuple: &'a (A, B)) -> &'a B {
	&tuple.1
}

pub fn keep_first_mut<'a, A, B>(tuple: &'a mut (A, B)) -> &'a mut A {
	&mut tuple.0
}

pub fn keep_second_mut<'a, A, B>(tuple: &'a mut (A, B)) -> &'a mut B {
	&mut tuple.1
}