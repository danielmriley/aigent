fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map(|(i, _)| i).unwrap_or(s.len())
}
fn main() {
    let mut s = String::from("hello");
    let idx = char_to_byte_idx(&s, 2);
    s.insert(idx, 'x');
    println!("{}", s);
}
