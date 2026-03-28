# Hyperparameters for 3 base agents + TTGS on various datasets
# pointmaze-giant-navigate-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.003 --agent.discount=0.995
# pointmaze-giant-navigate-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.0003 --agent.discount=0.995
# pointmaze-giant-navigate-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# pointmaze-giant-stitch-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=6 --threshold=12 --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.discount=0.995
# pointmaze-giant-stitch-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.0003 --agent.discount=0.995
# pointmaze-giant-stitch-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-giant-navigate-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3 --agent.discount=0.995
# antmaze-giant-navigate-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.003 --agent.discount=0.995
# antmaze-giant-navigate-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-giant-stitch-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.discount=0.995
# antmaze-giant-stitch-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.discount=0.995
# antmaze-giant-stitch-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0

# antmaze-large-explore-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
# antmaze-large-explore-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001
# antmaze-large-explore-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# humanoidmaze-giant-navigate-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-navigate-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-giant-navigate-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=36 --threshold=72 --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# humanoidmaze-giant-stitch-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-stitch-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-giant-stitch-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-antmaze-large-navigate-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-navigate-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-giant-navigate-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.alpha=0.3 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.003 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-navigate-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-giant-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-large-stitch-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-stitch-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-giant-stitch-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-antmaze-giant-stitch-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-giant-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0

# visual-antmaze-medium-explore-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-medium-explore-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-medium-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=10.0 --agent.low_actor_rep_grad=True --agent.low_alpha=10.0

# visual-antmaze-large-explore-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.001 --agent.batch_size=256 --agent.encoder=impala_small
# visual-antmaze-large-explore-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-antmaze-large-explore-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=10.0 --agent.low_actor_rep_grad=True --agent.low_alpha=10.0

# visual-humanoidmaze-medium-navigate-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/gciql.py --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-navigate-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-humanoidmaze-medium-navigate-v0 --train_steps=500000 --eval_episodes=50 --agent=agents/hiql.py --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100

# visual-humanoidmaze-medium-stitch-v0 (GCIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (QRL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.001 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small
# visual-humanoidmaze-medium-stitch-v0 (HIQL)
python main.py --subsample_ablt=random_points --random_size=4000 --tau=12 --threshold=24 --env_name=visual-humanoidmaze-medium-stitch-v0 --train_steps=500000 --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.batch_size=256 --agent.discount=0.995 --agent.encoder=impala_small --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=100
