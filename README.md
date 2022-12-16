# README

Install Python and make sure that the `python` command can be ran from your command line.

Install require python packages with:
```bash
pip install seaborn pandas matplotlib numpy scipy
```

Then run 
```bash
python sea_salt_experiment.py
```
to produce:

![](figures/berridge_juice_barchart_empty.png)

![](figures/berridge_salt_barchart_empty.png)

![](figures/empty_extinction_proper_sea_salt_rbtd_choices_empty.png)

![](figures/empty_extinction_proper_sea_salt_rbtd_reward_empty.png)

![](figures/no_homeostatic_proper_triple_sea_salt_rbtd_choices_2.png)

![](figures/no_homeostatic_proper_triple_sea_salt_rbtd_reward_2.png)

![](figures/sea_salt_extinction_homeostatic_choices.png)

![](figures/sea_salt_extinction_homeostatic_reward.png)

![](figures/sea_salt_extinction_rb_choices.png)

![](figures/sea_salt_extinction_rb_reward.png)

![](figures/sea_salt_extinction_td_choices.png)

![](figures/sea_salt_extinction_td_reward.png)

![](figures/sea_salt_extinction_td_reward.png)

Run 
```bash
python papageorgiou_results.py
``` 
to produce:

![](figures/cosyne_megaplot_3_empty.jpg)

![](figures/proper_PP_4A_4_empty_no_baseline.jpg)

Code for experiments and figures in "Reward Bases: Instanteous Reward Revaluation with Temporal Difference Learning". Paper can be found here: https://www.biorxiv.org/content/biorxiv/early/2022/04/14/2022.04.14.488361.full.pdf

Run
```bash
python room_task_experiments.py
```
to produce:

![](figures/interval_steps_maze_combined_1.png)

![](figures/learning_rate_maze_combined_6.png)

![](figures/room_task_quadruple_0.1.png)

## Further information

The files ``envs.py`` contains the environment classes for the various experiments and the ``learners.py`` file contains the classes for the reward basis, temporal difference, and successor representation agent
