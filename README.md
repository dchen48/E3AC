# E3AC
E3-Equivariant Actor-Critic Methods for Cooperative Multi-Agent Reinforcement Learning

1. Train E3AC on MPE:

```
cd E3AC/E3-MPE/maddpg
python main_vec_e3.py --scenario [simple_spread_n3,simple_spread_n6,simple_coop_push_n3,simple_coop_push_n6,simple_tag_n3,simple_tag_n6] --continuous --actor_type [MLP,gcn_max,segnn] --critic_type [MLP,segnn] --actor_lr [Please refer to the appendix] --critic_lr [Please refer to the appendix] --fixed_lr  --batch_size [Please refer to the appendix] --actor_clip_grad_norm 0.5 --cuda
```
2. Train E3AC on MuJoCo (multi-agent reacher):
```
cd E3AC/E3-MuJoCo/MA_Reacher
python train.py actor_type=[mlp,segnn] critic_type=[mlp,segnn] pixel_obs=false action_repeat=1 frame_stack=1 task=reacher_hard agent=ddpg_e3 lr=[Please refer to the appendix]
```
3. Train E3AC on MuJoCo (multi-agent hopper 3D):
```
cd E3AC/E3-MuJoCo/MA_Hopper_3D/src
python main.py --env_name hopper --morphologies hopper --exp_path ../results --config_path configs/3d.py --gpu 0 --custom_xml environments/3d_hoppers/3d_hopper_3_shin.xml --actor_type [mlp_v,segnn] --critic_type [mlp_v,segnn]
```


### Acknowledgement
The E3-MPE code is based on https://github.com/IouJenLiu/PIC.

The E3-MuJoCo is based on https://github.com/sahandrez/homomorphic_policy_gradient and https://github.com/alpc91/SGRL.
