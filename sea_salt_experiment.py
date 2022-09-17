# Dead Sea Salt Experiments

import numpy as np
from envs import *
from learners import *
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from scipy.signal import savgol_filter

### Reward Functions for the task ###
def salt_total_reward(env, state):
  if state[0] == 1:
    #print("STATE IS 1")
    return 1
  elif state[1] == 1:
    #print("STATE IS 2")
    return -10
  else:
    return 0
    #raise ValueError("STATE NOT VALID")

def salt_total_reward_after(env, state):
  if state[0] == 1:
    #print("STATE IS 1")
    return 1
  elif state[1] == 1:
    #print("STATE IS 2")
    return 10
  else:
    return 0
    #raise ValueError("STATE NOT VALID")

def salt_r1(env, state):
  if state[0] == 1:
    #print("STATE IS 1")
    return 1
  elif state[1] == 1:
    #print("STATE IS 2")
    return 0
  else:
    return 0
    #raise ValueError("STATE NOT VALID")


def salt_r2(env, state):
  if state[0] == 1:
    #print("STATE IS 1")
    return 0
  elif state[1] == 1:
    #print("STATE IS 2")
    return 1
  else:
    return 0
    #raise ValueError("STATE NOT VALID")

def salt_extinction_reward_fn(env, state):
  return 0


def plot_berridge_bar(vals, stds, labels,sname = "", clear_background = True,large_font = True, empty_plot = True,color = "blue"):
  xs = np.arange(len(labels))
  if not clear_background:
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  fig, ax = plt.subplots(figsize=(12,10))
  width = 0.7
  plt.bar(xs, vals, width, label = labels, alpha =0.8, yerr = stds, capsize = 4, color=color)
  #bar_td = ax.bar(xs + (width + 0.02), [TD_V_juice, TD_V_juice], width, label="Temporal Difference",alpha=0.8,yerr=[TD_STD_juice, TD_STD_juice],capsize=4)
  #bar_rb = ax.bar(xs , [V_normal_juice, V_depleted_juice], width, label="Reward Basis",alpha=0.8,yerr=[RB_std_juice, RB_std_juice],capsize=4)
  #bar_exp = ax.bar(xs - (width + 0.02) , [normal_juice, depleted_juice], width, label="Experiment",alpha=0.8,yerr=[normal_juice_std, depleted_juice_std],capsize=4)
  if large_font:
    ax.set_ylabel("Approach/Nibbles/Sniffs", fontsize=40)
  else:
    ax.set_ylabel("Approach/Nibbles/Sniffs", fontsize=28)
  ax.set_xlabel("", fontsize=28)
  if large_font:
    ax.set_title("Responses to Juice Lever",fontsize=45)
  else:
    ax.set_title("Responses to Juice Lever",fontsize=30)
  if empty_plot:
    ax.set_ylabel("", fontsize=28)
    ax.set_ylim(0,5)
    ax.set_yticks([])
    ax.set_title("")
  ax.set_xticks(xs)
  ax.set_xticklabels(labels)
  if large_font:
    ax.tick_params(axis='x', which='major', labelsize=35)
    ax.tick_params(axis='x', which='minor', labelsize=35)
    ax.tick_params(axis='y', which='major', labelsize=35)
    ax.tick_params(axis='y', which='minor', labelsize=35)
  else:
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='x', which='minor', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.tick_params(axis='y', which='minor', labelsize=20)
  if not empty_plot:
    ax.legend(fontsize=25)
  sns.despine(left=False,top=True, right=True, bottom=False)
  fig.tight_layout()
  if empty_plot:
    plt.savefig("figures/berridge_empty_" + sname + ".png", format="png")
  else:
    plt.savefig("figures/berridge_" + sname + ".png", format="png")
  plt.show()


def berridge_plot(learning_rate, beta, steps_per_reversal, N_agents=10, empty_plot = True, large_font=True, clear_background = True, group_by_type = False):
  env = SeaSaltExperiment()
  Vss = []
  for i in range(N_agents):
    agent = Reward_Basis_Learner(gamma,[salt_r1,salt_r2],env,learning_rate,beta,[1,0], random_policy=True)
    rs1, V1s, a1s = agent.interact(steps_per_reversal, return_actions=True)
    Vs = agent.Vs
    Vss.append(Vs)
  Vss = np.array(Vss)
  Vs = np.mean(Vss, axis=0)
  Vs_std = np.std(Vss, axis=0)
  print("VS std")
  print(Vs_std)
  print(Vss.shape)
  print(Vs)
  # TD learner
  Vss_td = []
  for i in range(N_agents):
    TD_agent = TD_Learner(gamma,salt_total_reward,env,learning_rate,beta,random_policy=True)
    rs1, V1s, a1s = TD_agent.interact(steps_per_reversal, return_actions = True)
    V = TD_agent.V
    Vss_td.append(V)
    
  Vss_td = np.array(Vss_td)
  Vs_td = np.mean(Vss_td,axis=0)
  Vs_td_std = np.std(Vss_td, axis=0)
    
  # berridge data from figure 3C
  normal_salt = 1
  normal_salt_std = 0.1
  normal_juice = 4.2
  normal_juice_std = 0.8
  depleted_salt = 3.3
  depleted_salt_std = 0.5
  depleted_juice = 4.5
  depleted_juice_std = 0.2
  
  # compute Vs
  V_juice = Vs[0,:]
  V_salt = Vs[1,:]
  # fitted alphas
  alpha_normal_juice = 4.2
  alpha_normal_salt = 1
  V_normal_juice = V_juice[0] * alpha_normal_juice + V_juice[1] * alpha_normal_salt
  V_normal_salt = V_salt[0] * alpha_normal_juice + V_salt[1] * alpha_normal_salt
  
  alpha_depleted_salt = 3.3
  alpha_depleted_juice = 4.5
  V_depleted_juice = V_juice[0] * alpha_depleted_juice + V_juice[1] * alpha_depleted_salt
  V_depleted_salt = V_salt[0] * alpha_depleted_juice + V_salt[1] * alpha_depleted_salt
  
  RB_std_juice = Vs_std[0,0]
  RB_std_salt = Vs_std[1,1]
  print("vs td ", Vs_td)
  
  TD_V_juice = Vs_td[0]
  TD_V_salt = Vs_td[1]
  w_juice = 4.2
  w_salt = -0.1 # to counteract negative value function
  TD_V_juice = w_juice * TD_V_juice
  TD_V_salt = w_salt * TD_V_salt
  print("TD STD: ",Vs_td_std)
  TD_STD_juice = Vs_td_std[0]
  TD_STD_salt = Vs_td_std[1]
  print(TD_STD_juice)
  print(TD_STD_salt)


  # rafal's other plotting idea
  if group_by_type:
    plot_berridge_bar([TD_V_juice, TD_V_juice, TD_V_salt, TD_V_salt],[TD_STD_juice, TD_STD_juice, TD_STD_salt, TD_STD_salt],labels=["Juice homeostasis", "Juice depleted","Salt Homeostasis","Salt Depleted"],sname="TD_bar", clear_background = clear_background,large_font = large_font,empty_plot = empty_plot,color="blue")
    plot_berridge_bar([V_normal_juice, V_depleted_juice, V_normal_salt, V_depleted_salt],[RB_std_juice, RB_std_juice, RB_std_salt, RB_std_salt],labels=["Juice homeostasis", "Juice depleted","Salt Homeostasis","Salt Depleted"], sname="RB_bar",clear_background = clear_background,large_font = large_font, empty_plot = empty_plot,color="orange")
    plot_berridge_bar([normal_juice, depleted_juice, normal_salt, depleted_salt],[normal_juice_std, depleted_juice_std, normal_salt_std, depleted_salt_std],labels=["Juice homeostasis", "Juice depleted","Salt Homeostasis","Salt Depleted"],sname="experiment_bar", clear_background = clear_background, large_font = large_font, empty_plot = empty_plot,color="green")
  
  else:
    # bar chart -- 2 bar charts one for juice one for salt.
    # juice plot
    labels = ["Homeostasis", "Sodium Depleted"]
    xs = np.arange(len(labels))
    if not clear_background:
      sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12,10))
    width = 0.2
    bar_td = ax.bar(xs + (width + 0.02), [TD_V_juice, TD_V_juice], width, label="Temporal Difference",alpha=0.8,yerr=[TD_STD_juice, TD_STD_juice],capsize=4)
    bar_rb = ax.bar(xs , [V_normal_juice, V_depleted_juice], width, label="Reward Basis",alpha=0.8,yerr=[RB_std_juice, RB_std_juice],capsize=4)
    bar_exp = ax.bar(xs - (width + 0.02) , [normal_juice, depleted_juice], width, label="Experiment",alpha=0.8,yerr=[normal_juice_std, depleted_juice_std],capsize=4)
    if large_font:
      ax.set_ylabel("Approach/Nibbles/Sniffs", fontsize=40)
    else:
      ax.set_ylabel("Approach/Nibbles/Sniffs", fontsize=28)
    ax.set_xlabel("", fontsize=28)
    if large_font:
      ax.set_title("Responses to Juice Lever",fontsize=45)
    else:
      ax.set_title("Responses to Juice Lever",fontsize=30)
    if empty_plot:
      ax.set_ylabel("", fontsize=28)
      ax.set_ylim(0,5)
      ax.set_yticks([])
      ax.set_title("")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    if large_font:
      ax.tick_params(axis='x', which='major', labelsize=35)
      ax.tick_params(axis='x', which='minor', labelsize=35)
      ax.tick_params(axis='y', which='major', labelsize=35)
      ax.tick_params(axis='y', which='minor', labelsize=35)
    else:
      ax.tick_params(axis='x', which='major', labelsize=25)
      ax.tick_params(axis='x', which='minor', labelsize=25)
      ax.tick_params(axis='y', which='major', labelsize=25)
      ax.tick_params(axis='y', which='minor', labelsize=25)
    if not empty_plot:
      ax.legend(fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    fig.tight_layout()
    if empty_plot:
      plt.savefig("figures/berridge_juice_barchart_empty.png", format="png")
    else:
      plt.savefig("figures/berridge_juice_barchart.png", format="png")
    plt.show()
    # salt plot
    labels = ["Homeostasis", "Sodium Depleted"]
    xs = np.arange(len(labels))
    if not clear_background:
      sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12,10))
    width = 0.2
    bar_td = ax.bar(xs + (width + 0.02) , [TD_V_salt, TD_V_salt], width, label="Temporal Difference",alpha=0.8,yerr=[TD_STD_salt, TD_STD_salt],capsize=4)
    bar_rb = ax.bar(xs , [V_normal_salt, V_depleted_salt], width, label="Reward Basis",alpha=0.8,yerr=[RB_std_salt, RB_std_salt],capsize=4)
    bar_exp = ax.bar(xs - (width + 0.02)  , [normal_salt, depleted_salt], width, label="Experiment",alpha=0.8,yerr=[normal_salt_std, depleted_salt_std],capsize=4)
    if large_font:
      ax.set_ylabel("Approach/Nibbles/Sniffs", fontsize=40)
    else:
      ax.set_ylabel("Approach/Nibbles/Sniffs", fontsize=28)
    if large_font:
      ax.set_xlabel("", fontsize=40)
      ax.set_title("Responses to Salt Lever",fontsize=40)
    else:
      ax.set_xlabel("", fontsize=28)
      ax.set_title("Responses to Salt Lever",fontsize=30)
    if empty_plot:
      ax.set_ylabel("", fontsize=28)
      ax.set_ylim(0,5)
      ax.set_yticks([])
      ax.set_title("")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    if large_font:
      ax.tick_params(axis='x', which='major', labelsize=35)
      ax.tick_params(axis='x', which='minor', labelsize=35)
      ax.tick_params(axis='y', which='major', labelsize=35)
      ax.tick_params(axis='y', which='minor', labelsize=35)
    else:
      ax.tick_params(axis='x', which='major', labelsize=25)
      ax.tick_params(axis='x', which='minor', labelsize=25)
      ax.tick_params(axis='y', which='major', labelsize=25)
      ax.tick_params(axis='y', which='minor', labelsize=25)
    if large_font:
      ax.legend(fontsize=35, loc="upper left")
    else:
      ax.legend(fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    fig.tight_layout()
    if empty_plot:
      plt.savefig("figures/berridge_salt_barchart_empty.png", format="png")
    else:
      plt.savefig("figures/berridge_salt_barchart.png", format="png")
    plt.show()

def sea_salt_reversal_protocol(agent, steps_per_reversal, r1,r2,RB_learner=False, salt_extinction = False):
  if RB_learner:
    agent.alphas = [1,-10]
    agent.rfuns = [r1, r2]
  else:
    agent.reward_function = r1
  rs1, V1s,a1s = agent.interact(steps_per_reversal, return_actions = True)
  # reversal
  if salt_extinction:
      print("SETTING VARIOUS REWARD FUNCTIONS")
      agent.reward_function = salt_extinction_reward_fn
      agent.rfuns = [salt_extinction_reward_fn,salt_extinction_reward_fn]
      agent.alphas = [-2,10]
      agent.kappa  = [-10,10]
  else:
    if RB_learner:
      agent.alphas = [1,10]
      agent.V = agent.compute_total_v(agent.alphas) # ensure that the value function is updated immediately
      #agent.env.termination_condition = agent.env.all_points_termination_condition
    else:
      agent.reward_function = r2
      if agent.homeostatic_agent:
        agent.kappa = [1,10]
  #agent.env.termination_condition = agent.env.all_points_termination_condition
  rs2,V2s,a2s = agent.interact(steps_per_reversal,return_actions=True)
  a1s = np.array(a1s)
  a2s = np.array(a2s)
  return np.concatenate((rs1, rs2),axis=0),np.concatenate((V1s, V2s),axis=0), np.concatenate((a1s, a2s), axis=0)



def plot_reward_figure(rs,rs_stds = None, sname=None,title="Reward During Reversal for Temporal Difference learner"):
  fig = plt.figure(figsize=(10,8))
  #fig, ax = plt.subplots(1,1)
  xs = np.arange(0,len(rs))
  sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  plt.plot(xs,rs,label="Reward",linewidth="2")
  if rs_stds is not None:
    plt.fill_between(xs, rs - rs_stds, rs + rs_stds, alpha=0.5)
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.xlabel("Timestep",fontsize=20)
  plt.ylabel("Total reward over each episode",fontsize=20)
  plt.title(title,fontsize=20)
  plt.axvline(len(xs)//2, color="green", linestyle="--",linewidth="1.5",label="Reversal Time")
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.legend(fontsize=20)
  if sname is not None:
    plt.savefig(sname)
  fig.tight_layout()
  plt.show()

def plot_actions_figure(actions, actions_stds = None, sname=None, title="Choice selected before and after reversal"):
  fig = plt.figure(figsize=(10,8))
  #fig, ax = plt.subplots(1,1)
  xs = np.arange(0,len(actions))
  sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  plt.plot(xs,actions,label="Choice",linewidth="2")
  if actions_stds is not None:
    plt.fill_between(xs, actions - actions_stds, actions + actions_stds, alpha=0.5)
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.xlabel("Timestep",fontsize=20)
  plt.ylabel("Choice at each time-step",fontsize=20)
  plt.title(title,fontsize=20)
  plt.axvline(len(actions)//2, color="green", linestyle="--",linewidth="1.5",label="Reversal Time")
  plt.yticks([0,1],["Sweet Juice","Sea Salt"],fontsize=13)
  plt.xticks(fontsize=15)
  plt.legend(fontsize=20)
  if sname is not None:
    plt.savefig(sname)

  fig.tight_layout()
  plt.show()

def plot_combined_figure(td_means, td_stds, rb_means, rb_stds, sname=None, title="Choice selected before and after reversal", ylabel_label="Mean Choice",label_yaxis=False, empty_plot = False, smooth=False, use_standard_error = False):
  fig = plt.figure(figsize=(12,10))
  xs = np.arange(0,len(td_means))
  if not empty_plot:
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  if use_standard_error:
    td_stds /= np.sqrt(len(td_stds))
    rb_stds /= np.sqrt(len(rb_stds))
  if smooth:
      td_stds = savgol_filter(td_stds, 101, 2)
      rd_stds = savgol_filter(rb_stds, 101, 2)
  plt.plot(xs,td_means,label="TD",linewidth="2")
  plt.fill_between(xs, td_means - td_stds, td_means + td_stds, alpha=0.3)
  plt.plot(xs,rb_means,label="RB",linewidth="2")
  plt.fill_between(xs, rb_means - rb_stds, rb_means + rb_stds, alpha=0.3)
  plt.axvline(len(td_means)//2, color="green", linestyle="--",linewidth="1.5",label="Reversal")
  sns.despine(left=False,top=True, right=True, bottom=False)
  if not empty_plot:
    plt.xlabel("Trial",fontsize=22)
    plt.ylabel(str(ylabel_label),fontsize=22)
    plt.title(title,fontsize=25)
    if label_yaxis:
      plt.yticks([0,1],["Juice","Salt"],fontsize=15,rotation=0)
    else:
      plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
  else:
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
  plt.legend(fontsize=28)
  if sname is not None:
    plt.savefig(sname)

  fig.tight_layout()
  plt.show()
  
def plot_triple_combined_figure(td_means, td_stds, rb_means, rb_stds, sr_means, sr_stds, sname="None", title="Average Reward before and after reversal", ylabel_label="Mean Reward",label_yaxis=False):
  fig = plt.figure(figsize=(12,10))
  xs = np.arange(0,len(td_means))
  sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  plt.plot(xs,td_means,label="TD",linewidth="2")
  plt.fill_between(xs, td_means - td_stds, td_means + td_stds, alpha=0.5)
  plt.plot(xs,rb_means,label="RB",linewidth="2")
  plt.fill_between(xs, rb_means - rb_stds, rb_means + rb_stds, alpha=0.5)
  plt.plot(xs,sr_means,label="Homeostatic",linewidth="2")
  plt.fill_between(xs, sr_means - sr_stds, sr_means + sr_stds, alpha=0.5)
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.xlabel("Timestep",fontsize=22)
  plt.ylabel(str(ylabel_label),fontsize=22)
  plt.title(title,fontsize=25)
  plt.axvline(len(td_means)//2, color="green", linestyle="--",linewidth="1.5",label="Reversal Time")
  if label_yaxis:
    plt.yticks([0,1],["Sweet Juice","Sea Salt"],fontsize=15)
  else:
    plt.yticks(fontsize=20)
  plt.xticks(fontsize=20)
  plt.legend(fontsize=22)
  if sname is not None:
    plt.savefig(sname)

  fig.tight_layout()
  plt.show()

def run_sea_salt_experiment(learning_rate, beta,gamma,steps_per_reversal,plot_results = False,salt_extinction=False):
  env = SeaSaltExperiment()
  TD_agent = TD_Learner(gamma,salt_total_reward,env,learning_rate,beta)
  RB_agent = Reward_Basis_Learner(gamma,[salt_r1,salt_r2],env,learning_rate,beta,[1,0])
  rs_rb,vs_rb, as_rb  = sea_salt_reversal_protocol(RB_agent,steps_per_reversal,salt_r1, salt_r2,RB_learner=True,salt_extinction = salt_extinction)
  rs_td,vs_td, as_td = sea_salt_reversal_protocol(TD_agent,steps_per_reversal,salt_total_reward, salt_total_reward_after,salt_extinction = salt_extinction)
  homeostatic_agent = Homeostatic_TD_Learner(gamma, salt_total_reward,env, learning_rate,beta,kappa=None, simulated_reward_update=True)
  rs_k, vs_k, as_k = sea_salt_reversal_protocol(homeostatic_agent,steps_per_reversal, salt_total_reward,salt_total_reward_after, salt_extinction= salt_extinction)
  if plot_results:
    plot_reward_figure(rs_rb,title="Reward During Reversal for Temporal Difference Learner")
    plot_reward_figure(rs_td,title="Reward During Reversal for Reward Basis Learner")
    plot_actions_figure(as_rb,title="Choices of Reward Basis Learner during Reversal")
    plot_actions_figure(as_td,title="Choices of Temporal Difference Learner during Reversal")
  return rs_rb, as_rb, rs_td, as_td, rs_k, as_k

def run_N_sea_salt_experiment(N_runs, learning_rate, beta, gamma, steps_per_reversal,salt_extinction=False,plot_combined_figure_flag = True,empty_plot = False,use_homeostatic=False):
    rs_rbs = []
    as_rbs = []
    rs_tds = []
    as_tds = []
    rs_ks = []
    as_ks = []
    for i in range(N_runs):
      rs_rb, as_rb, rs_td, as_td, rs_k, as_k = run_sea_salt_experiment(learning_rate, beta, gamma, steps_per_reversal,salt_extinction=salt_extinction)
      rs_rbs.append(rs_rb)
      as_rbs.append(as_rb)
      rs_tds.append(rs_td)
      as_tds.append(as_td)
      rs_ks.append(rs_k)
      as_ks.append(as_k)
    rs_rbs = np.array(rs_rbs)
    as_rbs = np.array(as_rbs)
    rs_tds = np.array(rs_tds)
    as_tds = np.array(as_tds)
    rs_ks = np.array(rs_ks)
    as_ks = np.array(as_ks)
    rs_rbs_mean = np.mean(rs_rbs, axis=0)
    rs_rbs_std = np.std(rs_rbs, axis=0)
    as_rbs_mean = np.mean(as_rbs, axis=0)
    as_rbs_std = np.std(as_rbs, axis=0)
    rs_tds_mean = np.mean(rs_tds, axis=0)
    rs_tds_std = np.std(rs_tds,axis=0)
    as_tds_mean = np.mean(as_tds,axis=0)
    as_tds_std = np.std(as_tds, axis=0)
    rs_ks_mean = np.mean(rs_ks, axis=0)
    rs_ks_std = np.std(rs_ks, axis=0)
    as_ks_mean = np.mean(as_ks, axis=0)
    as_ks_std = np.std(as_ks, axis=0)

    if salt_extinction:
      plot_reward_figure(rs_rbs_mean,rs_rbs_std,title="Reward During Reversal for Reward Basis Learner",sname="figures/sea_salt_extinction_rb_reward.png")
      plot_reward_figure(rs_tds_mean, rs_tds_std,title="Reward During Reversal for Termporal Difference Learner",sname="figures/sea_salt_extinction_td_reward.png")
      plot_actions_figure(as_rbs_mean, as_rbs_std,title="Choices of Reward Basis Learner during Reversal",sname="figures/sea_salt_extinction_rb_choices.png")
      plot_actions_figure(as_tds_mean, as_tds_std,title="Choices of Temporal Difference Learner during Reversal",sname="figures/sea_salt_extinction_td_choices.png")
      plot_reward_figure(rs_ks_mean, rs_ks_std,title="Reward During Reversal for Homeostatic Temporal Difference Learner",sname="figures/sea_salt_extinction_homeostatic_reward.png")
      plot_actions_figure(as_ks_mean, as_ks_std,title="Choices of Homeostatic Temporal Difference Learner During Reversal",sname="figures/sea_salt_extinction_homeostatic_choices.png")
  
    if plot_combined_figure_flag:
      plot_combined_figure(rs_tds_mean, rs_tds_std, rs_rbs_mean, rs_rbs_std, title="Reward for RB/TD Learner", sname="figures/empty_extinction_proper_sea_salt_rbtd_reward_empty.png",ylabel_label = "Mean Reward",empty_plot = empty_plot)
      plot_combined_figure(as_tds_mean, as_tds_std, as_rbs_mean, as_rbs_std, title="Choices of RB/TD Learner", sname="figures/empty_extinction_proper_sea_salt_rbtd_choices_empty.png",label_yaxis=True, empty_plot = empty_plot,smooth=True)
      # triple for homeostatic
      if use_homeostatic:
        plot_triple_combined_figure(rs_tds_mean, rs_tds_std, rs_rbs_mean, rs_rbs_std,rs_ks_mean, rs_ks_std, title="Reward for RB/TD Learner", sname="figures/no_homeostatic_proper_triple_sea_salt_rbtd_reward_2.png",ylabel_label = "Mean Reward")
        plot_triple_combined_figure(as_tds_mean, as_tds_std, as_rbs_mean, as_rbs_std,as_ks_mean, as_ks_std, title="Choices of RB/TD Learner", sname="figures/no_homeostatic_proper_triple_sea_salt_rbtd_choices_2.png",label_yaxis=True)
    else:
      plot_reward_figure(rs_rbs_mean,rs_rbs_std,title="Reward During Reversal for Reward Basis Learner",sname="figures/sea_salt_rb_reward.png")
      plot_reward_figure(rs_tds_mean, rs_tds_std,title="Reward During Reversal for Termporal Difference Learner",sname="figures/sea_salt_td_reward.png")
      plot_actions_figure(as_rbs_mean, as_rbs_std,title="Choices of Reward Basis Learner during Reversal",sname="figures/sea_salt_rb_choices.png")
      plot_actions_figure(as_tds_mean, as_tds_std,title="Choices of Temporal Difference Learner during Reversal",sname="figures/sea_salt_td_choices.png")
      plot_reward_figure(rs_ks_mean, rs_ks_std,title="Reward During Reversal for Homeostatic Temporal Difference Learner",sname="figures/sea_salt_homeostatic_reward.png")
      plot_actions_figure(as_ks_mean, as_ks_std,title="Choices of Homeostatic Temporal Difference Learner During Reversal",sname="figures/sea_salt_homeostatic_choices.png")


if __name__ == '__main__':
  if not os.path.exists("figures/"):
    os.makedirs("figures/")
  learning_rate = 0.1
  beta = 1
  beta_random_exploration = 0.2
  gamma = 1
  steps_per_reversal = 10
  GROUP_BY_TYPE = False # whether to plot berridge bar chart expts by type or not
  #run_sea_salt_experiment(learning_rate, beta, gamma, steps_per_reversal)
  salt_extinction = True

  berridge_plot(learning_rate, beta, steps_per_reversal,large_font=False, group_by_type=GROUP_BY_TYPE)
  run_N_sea_salt_experiment(20, learning_rate, beta_random_exploration, gamma, steps_per_reversal,salt_extinction=salt_extinction,empty_plot = True, use_homeostatic=True)