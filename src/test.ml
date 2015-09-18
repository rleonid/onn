
let td_vd_ref = ref None

let do_it ~batch_size ~hidden_layers ~epochs ~learning_rate =
  let td, vd =
    match !td_vd_ref with
    | None ->
      let d = Load_mnist.data `Train in
      let s = Onn.split_validation 10000 d in
      td_vd_ref := Some s;
      s
    | Some p -> p
  in
  let t = Mnist.nn hidden_layers in
  Onn.sgd Mnist.input_size td
    ~epochs
    ~batch_size
    ~learning_rate
    ~report:(fun t ->
      let (c,d) = Onn.report_accuracy Mnist.input_size vd t in
      Printf.printf "%d out of %d\n%!" c d)
    Onn.rmse_cdf
    t;
  t

let simple_t () = do_it ~batch_size:50 ~hidden_layers:20 ~epochs:30 ~learning_rate:0.3

let () =
  let t = do_it ~batch_size:40 ~hidden_layers:100 ~epochs:30 ~learning_rate:0.25 in
  let test_d = Load_mnist.data `Test in
  let (c,t) = Onn.report_accuracy Mnist.input_size test_d t in
  Printf.printf "%d out of %d on test data.\n" c t
