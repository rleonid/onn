
(* To inspect overfitting in the absence of regularization. *)
open Lacaml.D

let () =
  let t =
    Onn.do_it
      ~training_size:1000
      ~cache:true
      ~batch_size:10
      ~num_hidden_nodes:30
      ~epochs:2000
      ~learning_rate:0.45
      ~lambda:0.01
      ~training_perf:(fun td t ->
          let x       = lacpy ~m:Mnist.input_size td in
          let y_hat_m = Onn.eval_m t (Onn.batch_eda (Mat.dim2 td) t) x in
          let e, _    =
            Mat.fold_cols (fun (e_s,i) y_hat ->
              let y = copy ~ofsx:(Mnist.input_size + 1) (Mat.col td i) in
              let e = Onn.cross_entropy_cost y y_hat |> Vec.sum in
              (e_s +. e), i + 1)
                (0.0, 1)
                y_hat_m
          in
          Printf.printf "training cost: %f\n" e)
      Onn.cross_entropy_cdf
  in
  let test_d = Load_mnist.data `Test in
  let (c,t) = Onn.report_accuracy Mnist.input_size test_d t in
  Printf.printf "%d out of %d on test data.\n" c t
