
let simple_t () = Onn.do_it ~batch_size:50 ~hidden_layers:20 ~epochs:30 ~learning_rate:0.3

let () =
  let t = Onn.do_it ~cache:true ~iterative:false ~batch_size:10 ~hidden_layers:100 ~epochs:30 ~learning_rate:3.0 in
  let test_d = Load_mnist.data `Test in
  let (c,t) = Onn.report_accuracy Mnist.input_size test_d t in
  Printf.printf "%d out of %d on test data.\n" c t
