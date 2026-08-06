python scripts/export_robotwin_dino_student_action_onnx.py \
  --ckpt /wangx1211/zwl/FastWAM_copy/runs/robotwin_dino_student_action_3cam_384_proprio_concatpred_actionconcat_wancond_detachactstudent_detachwan_textlinearconcat_vith16plus_1e-4_cosmos_reprpool2x2_deepsup/2026-07-10_12-41-22/checkpoints/weights/step_094925.pt \
  --action-onnx outputs/tmp/action.onnx \
  --student-onnx outputs/tmp/student.onnx \
  --action-engine outputs/tmp/action.trt \
  --student-engine outputs/tmp/student.trt \
  --trtexec /wangx1211/TensorRT-11.0.0.114/bin/trtexec \
  --action-horizon 32