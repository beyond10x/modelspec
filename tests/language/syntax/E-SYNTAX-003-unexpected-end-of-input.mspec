// expect E-SYNTAX-003 10:1 unexpected end of input
model M {
    param w: Tensor<f32, [2, 1]> = zeros();
    fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> {
        return matmul(x, w);
    }
}
train T {
    model = M;
