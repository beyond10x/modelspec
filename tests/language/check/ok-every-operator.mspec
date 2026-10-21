// expect ok
train EveryOp {
    model = EveryOperator;
    input x: Tensor<f32, [8, 4]>;
    input z: Tensor<f32, [8, 1]>;
    target y: Tensor<f32, [8, 1]>;
    loss = add(mse(model.forward(x, z), y), mul(0.5, sum(z)));
    optimizer = sgd(lr = 1.0e-3);
    steps = 3;
    seed = 0;
    device = cpu;
}

model EveryOperator {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param v: Tensor<f32, [1, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>, z: Tensor<f32, [8, 1]>) -> Tensor<f32, [8, 1]> {
        let h = relu(matmul(x, w));
        let t = transpose(v);
        let s = add(h, t);
        let d = sub(s, mul(h, 0.5));
        let m = mul(mean(x), d);
        let q = mul(sum(w), m);
        return add(q, z);
    }

    fn unused() -> Tensor<f32, []> {
        return 2.5;
    }
}
