docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v /home/ngoctruong:/root/workspace  \
rocm/composable_kernel:ck_ub20.04_rocm6.0          \
/bin/bash