
let simple_t () = Onn.do_it ~batch_size:50 ~num_hidden_nodes:20 ~epochs:30 ~learning_rate:0.3

let () =
  let t =
    Onn.do_it
      ~batch_size:10
      ~num_hidden_nodes:100
      ~epochs:30
      ~learning_rate:0.5
      ~lambda:5.0
      Onn.cross_entropy_cdf
  in
  let test_d = Load_mnist.data `Test in
  let (c,t) = Onn.report_accuracy Mnist.input_size test_d t in
  Printf.printf "%d out of %d on test data.\n" c t
