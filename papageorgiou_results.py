# Quick script demonstrating our method can predict experimental results from Papageorgiou et al 2016 (https://www.sciencedirect.com/science/article/pii/S221112471630287X)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

# matplotlib.rcParams.update({'errorbar.capsize': 3})
sns.set_theme('talk', font_scale=1.2)


def plot_pp_1C():
    # plot figure matching qualitative results of papageorgiou paper figure 1C
    labels = ["MORE Valued", "MORE Devalued"]
    food_means = [1, 0.2]
    sucrose_means = [0.9, 0.1]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, food_means, width, label='Food')
    bar2 = ax.bar(x + width/2, sucrose_means, width, label='Sucrose')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Relative Dopamine Level',
                  # fontsize=25
                  )
    ax.set_title('Average Dopamine Levels for MORE received',
                 # fontsize=25
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='x', which='minor', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.tick_params(axis='y', which='minor', labelsize=10)
    # ax.legend(fontsize=22)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("PP_1C", format="pdf")
    # plt.show(block=False)


def plot_rb_1C():
    # get results from pp 1c using reward bases. Key thing is we assume basically food/sucrose is expected. The other reward type is NOT expected -- i.e delta = 0. # also assume that food is enjoyed slightly more than sucrose
    # generally total dopamine is sum of reward contributioned weighted by value into the NAcc
    # setup parameters
    food_MORE_delta = 1
    food_delta = 0.1
    food_valued_alpha = 1
    food_devalued_alpha = 0.2
    sucrose_MORE_delta = 1
    sucrose__delta = 0.1
    sucrose_valued_alpha = 0.9
    sucrose_devalued_alpha = 0.1
    food_valued_MORE = food_valued_alpha * food_MORE_delta
    food_devalued_MORE = food_devalued_alpha * food_MORE_delta
    sucrose_valued_MORE = sucrose_valued_alpha * sucrose_MORE_delta
    sucrose_devalued_MORE = sucrose_devalued_alpha * sucrose_MORE_delta

    labels = ["MORE Valued", "MORE Devalued"]
    food_means = [food_valued_MORE, food_devalued_MORE]
    sucrose_means = [sucrose_valued_MORE, sucrose_devalued_MORE]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, food_means, width, label='Food')
    bar2 = ax.bar(x + width/2, sucrose_means, width, label='Sucrose')

    ax.set_ylabel('Relative Dopamine Level',
                  # fontsize=25
                  )
    ax.set_title('Dopamine Levels for MORE received',
                 # fontsize=25
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='x', which='minor', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.tick_params(axis='y', which='minor', labelsize=10)
    # ax.legend(fontsize=22)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("RB_1C", format="pdf")
    # plt.show(block=False)


def plot_pp_1D():
    # Plotting the qualitative shape of the results in the Papageorgiou paper for Figure 1D
    labels = ["SWITCH", "STAY"]
    food_means = [0.9, 0.5]
    sucrose_means = [0.4, 0.3]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, food_means, width, label='Food')
    bar2 = ax.bar(x + width/2, sucrose_means, width, label='Sucrose')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Relative Dopamine Level',
                  # fontsize=25
                  )
    ax.set_title('Dopamine levels for SWITCH vs STAY',
                 # fontsize=25
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.tick_params(axis='x', which='major', labelsize=20)
    # ax.tick_params(axis='x', which='minor', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=13)
    # ax.tick_params(axis='y', which='minor', labelsize=10)
    # ax.legend(fontsize=22)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("PP_1D", format="pdf")
    # plt.show(block=False)


def plot_rb_1D():
    # This function plots the predicted model results for Figure 1D in the Papageorgiou paper. Specifically, the SWITCH vs STAY results where the model diverges from the qualitative results in the paper for the DEVALUED-STAY condition
    alpha_food_valued = 0.9
    alpha_sucrose_devalued = 0.5
    delta_switch = 0.7
    delta_expected = 0.3
    baseline = 0.5

    da_food_switch = baseline + \
        (alpha_food_valued * delta_switch) + \
        (alpha_sucrose_devalued * -1 * delta_switch)
    da_food_expected = baseline + \
        (alpha_food_valued * delta_expected) + \
        (alpha_sucrose_devalued * -1 * delta_expected)
    da_sucrose_switch = baseline + \
        (alpha_food_valued * -1 * delta_switch) + \
        (alpha_sucrose_devalued * delta_switch)
    da_sucrose_expected = baseline + \
        (alpha_food_valued * -1 * delta_expected) + \
        (alpha_sucrose_devalued * delta_expected)

    labels = ["SWITCH", "STAY"]
    food_means = [da_food_switch, da_food_expected]
    sucrose_means = [da_sucrose_switch, da_sucrose_expected]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, food_means, width, label='Food')
    bar2 = ax.bar(x + width/2, sucrose_means, width, label='Sucrose')

    plt.axhline(baseline, color="grey", linestyle="--",
                linewidth="1.5", label="Baseline Dopamine")

    ax.set_ylabel('Dopamine Level',
                  # fontsize=25
                  )
    ax.set_title('Dopamine Levels for SWITCH vs STAY',
                 # fontsize=25
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.tick_params(axis='x', which='major',
    # labelsize=20
    # )
    # ax.tick_params(axis='x', which='minor',
    # labelsize=10
    # )
    # ax.tick_params(axis='y', which='major',
    # labelsize=13
    # )
    # ax.tick_params(axis='y', which='minor',
    # labelsize=10
    # )
    # ax.legend(fontsize=22)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("RB_1D", format="pdf")
    # plt.show(block=False)


def plot_pp_2AB():
    # this function plots the qualitative results from Figure 2 A and B of the Papageorgiou paper
    labels = ["Valued", "Devalued"]
    food_means = [0.9, 0.4]
    sucrose_means = [0.9, 0.4]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, food_means, width, label='Food')
    bar2 = ax.bar(x + width/2, sucrose_means, width, label='Sucrose')

    ax.set_ylabel('Dopamine Level',
                  # fontsize=25
                  )
    ax.set_title(
        'Dopamine Levels for Valued/Devalued at Cue Onset',
        # fontsize=25
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.tick_params(axis='x', which='major', labelsize=20)
    # ax.tick_params(axis='x', which='minor', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=13)
    # ax.tick_params(axis='y', which='minor', labelsize=10)
    # ax.legend(fontsize=22)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("PP_2AB", format="pdf")
    # plt.show(block=False)


def plot_rb_2AB():
    # this function plots the reward basis model predictions of the condition i Figure 2 A and B of the Papageorgiou paper. Specifically, it predicts the general level of dopamine release at cue onset for the valued and devalued rewards
    baseline = 0.5
    delta_food_food_received = 0.5
    delta_sucrose_sucrose_received = 0.5
    delta_food_sucorse_received = -0.5
    delta_sucrose_food_received = -0.5
    alpha_food_valued = 0.9
    alpha_food_devalued = 0.4
    alpha_sucrose_valued = 0.9
    alpha_sucrose_devalued = 0.4

    food_received_valued = baseline + (alpha_food_valued * delta_food_food_received) + (
        alpha_sucrose_devalued * delta_sucrose_food_received)
    food_received_devalued = baseline + \
        (alpha_food_devalued * delta_food_food_received) + \
        (alpha_sucrose_valued * delta_sucrose_food_received)
    sucrose_received_valued = baseline + (alpha_food_devalued * delta_food_sucorse_received) + (
        alpha_sucrose_valued * delta_sucrose_sucrose_received)
    sucrose_received_devalued = baseline + (alpha_food_valued * delta_food_sucorse_received) + (
        alpha_sucrose_devalued * delta_sucrose_sucrose_received)

    labels = ["Valued", "Devalued"]
    food_means = [food_received_valued, food_received_devalued]
    sucrose_means = [sucrose_received_valued, sucrose_received_devalued]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, food_means, width, label='Food')
    bar2 = ax.bar(x + width/2, sucrose_means, width, label='Sucrose')

    plt.axhline(baseline, color="grey", linestyle="--",
                linewidth="1.5", label="Baseline Dopamine")

    ax.set_ylabel('Dopamine Level',
                  # fontsize=25
                  )
    ax.set_title(
        'Dopamine Levels for Valued/Devalued at Cue Onset',
        # fontsize=25
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.tick_params(axis='x', which='major', labelsize=20)
    # ax.tick_params(axis='x', which='minor', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=13)
    # ax.tick_params(axis='y', which='minor', labelsize=10)
    # ax.legend(fontsize=22)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("RB_2AB", format="pdf")
    # plt.show(block=False)


def train_agent_papageorgiou_exp(N_steps=500, lr=0.1):
    V_food = np.zeros(2)  # init V to 0
    V_water = np.zeros(2)
    a_food = 1
    a_water = 1
    s1_p1 = 0.9
    s1_p2 = 0.1
    s1_ps = np.array([s1_p1, s1_p2])
    s2_p1 = 0.1
    s2_p2 = 0.9
    s2_ps = np.array([s2_p1, s2_p2])
    r = 1
    EPSILON = 1e-5
    for n in range(N_steps):
        # compute total V and normalize
        V = (a_food * V_food) + (a_water * V_water)
        print(V)
        V = softmax(V)
        print(V)
        a = np.random.choice([0, 1], p=V)
        print("A", a)
        if a == 0:  # state 1
            r_idx = int(np.random.choice([0, 1], p=s1_ps))
            if r_idx == 0:
                # received food
                V_food[0] += lr * (r - V_food[0])
                V_water[0] += lr * (0 - V_water[0])
            if r_idx == 1:
                # received water
                V_food[0] += lr * (0 - V_food[0])
                V_water[0] += lr * (1 - V_water[0])
        if a == 1:
            r_idx = int(np.random.choice([0, 1], p=s2_ps))
            if r_idx == 0:
                # received food
                V_food[1] += lr * (r - V_food[1])
                V_water[1] += lr * (0 - V_water[1])
            if r_idx == 1:
                # received water
                V_food[1] += lr * (0 - V_food[1])
                V_water[1] += lr * (1 - V_water[1])
        print("V_food ", V_food)
        print("V_water", V_water)
    return V_food, V_water


def pp_2A(V_foods, V_waters):
    s1_p1 = 0.9
    s1_p2 = 0.1
    s2_p1 = 0.1
    s2_p2 = 0.9
    baseline = 0.5
    a_food = 0.6
    a_water = 0.49
    deltas_food_received = np.zeros((len(V_foods), 2))
    deltas_water_received = np.zeros((len(V_foods), 2))
    PP_food = 0.6
    PP_water = 0.4
    for i in range(len(V_foods)):
        # assuming having received food
        V_food = V_foods[i, :]
        V_water = V_waters[i, :]

        delta_food = (s1_p1 * V_food[0]) + (s2_p1 * V_food[1])
        delta_water = (s1_p2 * V_water[0]) + (s2_p2 * V_water[1])
        deltas_food_received[i, :] = np.array([delta_food, -delta_water])

        # assuming having received water!
        delta2_food = (s1_p2 * V_food[0]) + (s2_p2 * V_food[1])
        delta2_water = (s1_p1 * V_water[0]) + (s2_p1 * V_water[1])
        deltas_water_received[i, :] = np.array([-delta_food, delta_water])

    print("DELTAS: ", deltas_food_received)
    total_ds_food = []
    total_ds_water = []
    for i in range(len(V_foods)):
        total_ds_food.append(
            (a_food * deltas_food_received[i, 0]) + (a_water * deltas_food_received[i, 1]))
        total_ds_water.append(
            (a_food * deltas_water_received[i, 0]) + (a_water * deltas_water_received[i, 1]))
    print(total_ds_food)
    print(total_ds_water)
    mean_ds_food = np.mean(np.array(total_ds_food)) + baseline
    mean_ds_water = np.mean(np.array(total_ds_water)) + baseline
    std_ds_food = np.std(np.array(total_ds_food))
    std_ds_water = np.std(np.array(total_ds_water))
    print(mean_ds_food)
    print(mean_ds_water)
    labels = ["Food (Valued)", "Sucrose (Devalued)"]
    PP_means = [PP_food, PP_water]
    sim_means = [mean_ds_food, mean_ds_water]
    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/1.5, PP_means, width,
                  label='Experiment')
    bar2 = ax.bar(x + width/1.5, sim_means, width,
                  yerr=[std_ds_food, std_ds_water], label='Simulation')

    ax.set_ylabel('Relative Dopamine (DA) Level',
                  # fontsize=28
                  )
    ax.set_title('Dopamine for Valued/Devalued at Cue Onset',
                 # fontsize=30
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.tick_params(axis='x', which='major', labelsize=25)
    # ax.tick_params(axis='x', which='minor', labelsize=25)
    # ax.tick_params(axis='y', which='major', labelsize=25)
    # ax.tick_params(axis='y', which='minor', labelsize=25)
    # ax.legend(fontsize=25)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("figures/proper_PP_2AB_2.pdf", format="pdf")
    # plt.show(block=False)


def PP_2C(V_foods, V_waters, average_conditions=False, return_data=False, input_baselining=True):
    s1_p1 = 0.9
    s1_p2 = 0.1
    s2_p1 = 0.1
    s2_p2 = 0.9
    if input_baselining:
        baseline = 0.2
        a_food_valued = 0.35
        a_food_devalued = 0.1
        a_water_valued = 0.35
        a_water_devalued = 0.1
        PP_food_valued = 0.55
        PP_food_devalued = 0.1
        PP_sucrose_valued = 0.5
        PP_sucrose_devalued = -0.05
    else:
        baseline = 0.4
        a_food_valued = 0.35
        a_food_devalued = 0.06
        a_water_valued = 0.33
        a_water_devalued = 0.01
        PP_food_valued = 0.8
        PP_food_devalued = 0.2
        PP_sucrose_valued = 0.7
        PP_sucrose_devalued = 0.1
    deltas_food_more_food = []
    deltas_food_more_water = []
    deltas_water_more_food = []
    deltas_water_more_water = []
    r_more = 2
    for i in range(len(V_foods)):
        V_food = V_foods[i, :]
        V_water = V_waters[i, :]
        deltas_food_more_food.append(
            (s1_p1 * (r_more - V_food[0])) + (s2_p1 * (r_more - V_food[1])))
        deltas_water_more_food.append(
            (s1_p2 * (0 - V_water[0])) + (s2_p2 * (0 - V_water[1])))
        deltas_food_more_water.append(
            (s1_p1 * (0 - V_food[0])) + (s2_p1 * (0 - V_food[1])))
        deltas_water_more_water.append(
            (s1_p2 * (r_more - V_water[0])) + (s2_p2 * (r_more - V_water[1])))

    ds_more_food_valued = []
    ds_more_food_devalued = []
    ds_more_water_valued = []
    ds_more_water_devalued = []
    for i in range(len(V_foods)):
        ds_more_food_valued.append(
            (a_food_valued * deltas_food_more_food[i]) + (a_water_devalued * deltas_water_more_food[i]))
        ds_more_food_devalued.append(
            (a_food_devalued * deltas_food_more_food[i]) + (a_water_valued * deltas_water_more_food[i]))
        ds_more_water_valued.append(
            (a_food_devalued * deltas_food_more_water[i]) + (a_water_valued * deltas_water_more_water[i]))
        ds_more_water_devalued.append(
            (a_food_valued * deltas_food_more_water[i]) + (a_water_devalued * deltas_water_more_water[i]))

    ds_more_food_valued = np.array(ds_more_food_valued)
    ds_more_food_devalued = np.array(ds_more_food_devalued)
    ds_more_water_valued = np.array(ds_more_water_valued)
    ds_more_water_devalued = np.array(ds_more_water_devalued)
    mean_food_valued = np.mean(ds_more_food_valued) + baseline
    std_food_valued = np.std(ds_more_food_valued)
    mean_food_devalued = np.mean(ds_more_food_devalued) + baseline
    std_food_devalued = np.std(ds_more_food_devalued)
    mean_water_valued = np.mean(ds_more_water_valued) + baseline
    std_water_valued = np.std(ds_more_water_valued)
    mean_water_devalued = np.mean(ds_more_water_devalued) + baseline
    std_water_devalued = np.std(ds_more_water_devalued)

    # flag as to whether to break out food vs sucrose separately or average them together as valued/devalued
    if average_conditions:
        labels = ["MORE Valued", "MORE Devalued"]
        PP_valued = (PP_food_valued + PP_sucrose_valued) / 2
        PP_devalued = (PP_food_devalued + PP_sucrose_devalued) / 2
        PP_means = [PP_valued, PP_devalued]
        sim_means = [(mean_food_valued + mean_water_valued)/2,
                     (mean_food_devalued + mean_water_devalued) / 2]
        sim_stds = [(std_food_valued + std_water_valued),
                    (std_food_devalued + std_water_devalued)]

    else:
        labels = ["MORE Food Valued", "MORE Food Devalued",
                  "MORE Sucrose Valued", "MORE Sucrose Devalued"]
        PP_means = [PP_food_valued, PP_food_devalued,
                    PP_sucrose_valued, PP_sucrose_devalued]
        sim_means = [mean_food_valued, mean_food_devalued,
                     mean_water_valued, mean_water_devalued]
        sim_stds = [std_food_valued, std_food_devalued,
                    std_water_valued, std_water_devalued]

    if return_data:
        return PP_means, sim_means, sim_stds

    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/1.5, PP_means, width,
                  label='Experiment')
    bar2 = ax.bar(x + width/1.5, sim_means, width,
                  yerr=sim_stds, label='Simulation')

    ax.set_ylabel('Relative Dopamine (DA) Level',
                  # fontsize=28
                  )
    ax.set_title('Dopamine for Valued/Devalued when MORE',
                 # fontsize=30
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.tick_params(axis='x', which='major', labelsize=25)
    # ax.tick_params(axis='x', which='minor', labelsize=25)
    # plt.xticks(rotation=10)
    # ax.tick_params(axis='y', which='major', labelsize=25)
    # ax.tick_params(axis='y', which='minor', labelsize=25)
    # ax.legend(fontsize=25)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    if average_conditions:
        plt.savefig("figures/proper_PP_2C_averaged_2.pdf")
    else:
        plt.savefig("figures/proper_PP_2C_2_2.pdf", format="pdf")
    # plt.show(block=False)


def PP_2D(V_foods, V_waters, return_data=False, input_baselineing=True):
    if input_baselineing:
        PP_neg_baseline = 0.62
        PP_food_switch = 0.7 - PP_neg_baseline
        PP_sucrose_switch = 0.4 - PP_neg_baseline  # 0.5
        PP_food_expected = 0.5 - PP_neg_baseline
        PP_sucrose_expected = 0.45 - PP_neg_baseline
        baseline = -0.15  # 0.45
        a_food_valued = 0.55
        a_water_devalued = 0.3
    else:
        PP_food_switch = 0.7
        PP_sucrose_switch = 0.5
        PP_food_expected = 0.45
        PP_sucrose_expected = 0.4
        baseline = 0.45
        a_food_valued = 0.55
        a_water_devalued = 0.3
    deltas_food_food_switch = []
    deltas_food_food_expected = []
    deltas_water_food_switch = []
    deltas_water_food_expected = []
    deltas_water_water_switch = []
    deltas_water_water_expected = []
    deltas_food_water_switch = []
    deltas_food_water_expected = []

    deltas_food_switch = []
    deltas_water_switch = []
    deltas_food_expected = []
    deltas_water_expected = []
    for i in range(len(V_foods)):
        V_food = V_foods[i, :]
        V_water = V_waters[i, :]

        deltas_food_food_switch.append(1 - V_food[1])
        deltas_food_food_expected.append(1 - V_food[0])
        deltas_water_food_switch.append(0 - V_water[1])
        deltas_water_food_expected.append(0 - V_water[0])
        deltas_water_water_switch.append(1 - V_water[0])
        deltas_water_water_expected.append(1 - V_water[1])
        deltas_food_water_switch.append(0 - V_food[0])
        deltas_food_water_expected.append(0 - V_food[1])

        deltas_food_switch.append(
            (a_food_valued * deltas_food_food_switch[i]) + (a_water_devalued * deltas_water_food_switch[i]))
        deltas_water_switch.append((a_food_valued * deltas_food_water_switch[i]) + (
            a_water_devalued * deltas_water_water_switch[i]))
        deltas_food_expected.append((a_food_valued * deltas_food_food_expected[i]) + (
            a_water_devalued * deltas_water_food_expected[i]))
        deltas_water_expected.append((a_food_valued * deltas_food_water_expected[i]) + (
            a_water_devalued * deltas_water_water_expected[i]))

    deltas_food_switch = np.array(deltas_food_switch)
    deltas_water_switch = np.array(deltas_water_switch)
    deltas_food_expected = np.array(deltas_food_expected)
    deltas_water_expected = np.array(deltas_water_expected)

    mean_delta_food_switch = np.mean(deltas_food_switch) + baseline
    mean_delta_water_switch = np.mean(deltas_water_switch) + baseline
    mean_delta_food_expected = np.mean(deltas_food_expected) + baseline
    mean_delta_water_expected = np.mean(deltas_water_expected) + baseline

    std_delta_food_switch = np.std(deltas_food_switch)
    std_delta_water_switch = np.std(deltas_water_switch)
    std_delta_food_expected = np.std(deltas_food_expected)
    std_delta_water_expected = np.std(deltas_water_expected)

    labels = ["Valued SWITCH", "Devalued SWITCH",
              "Valued Expected", "Devalued Expected"]

    PP_means = [PP_food_switch, PP_sucrose_switch,
                PP_food_expected, PP_sucrose_expected]
    sim_means = [mean_delta_food_switch, mean_delta_water_switch,
                 mean_delta_food_expected, mean_delta_water_expected]
    sim_stds = [std_delta_food_switch, std_delta_water_switch,
                std_delta_food_expected, std_delta_water_expected]

    if return_data:
        return PP_means, sim_means, sim_stds

    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/1.5, PP_means, width,
                  label='Experiment')
    bar2 = ax.bar(x + width/1.5, sim_means, width,
                  yerr=sim_stds, label='Simulation')

    ax.set_ylabel('Relative Dopamine (DA) Level',
                  # fontsize=28
                  )
    ax.set_title('Dopamine after Reward Onset Comparison',
                 # fontsize=30
                 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', which='major', labelsize=25)
    # ax.tick_params(axis='x', which='minor', labelsize=25)
    plt.xticks(rotation=10)
    # ax.tick_params(axis='y', which='major', labelsize=25)
    # ax.tick_params(axis='y', which='minor', labelsize=25)
    # ax.legend(fontsize=24)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    plt.savefig("figures/proper_PP_2D_4.pdf", format="pdf")
    # plt.show(block=False)


def cosyne_combined_megaplot(V_foods, V_waters, empty_plot=False):
    MORE_PP_means, MORE_sim_means, MORE_sim_stds = PP_2C(
        V_foods, V_waters, return_data=True, average_conditions=True)
    SWITCH_PP_means, SWITCH_sim_means, SWITCH_sim_stds = PP_2D(
        V_foods, V_waters, return_data=True)
    labels = ["Valued Expected", "Devalued Expected", "MORE Valued",
              "MORE Devalued", "SWITCH Valued", "SWITCH Devalued"]
    PP_means = [SWITCH_PP_means[2], SWITCH_PP_means[3], MORE_PP_means[0],
                MORE_PP_means[1], SWITCH_PP_means[0], SWITCH_PP_means[1]]
    sim_means = [SWITCH_sim_means[2], SWITCH_sim_means[3], MORE_sim_means[0],
                 MORE_sim_means[1], SWITCH_sim_means[0], SWITCH_sim_means[1]]
    sim_stds = [SWITCH_sim_stds[2], SWITCH_sim_stds[3], MORE_sim_stds[0],
                MORE_sim_stds[1], SWITCH_sim_stds[0], SWITCH_sim_stds[1]]

    # if not empty_plot:
    #     sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/1.5, PP_means, width,
                  label='Experiment')
    bar2 = ax.bar(x + width/1.5, sim_means, width,
                  yerr=sim_stds, label='Simulation')
    if not empty_plot:
        ax.set_ylabel('Relative Dopamine',
                      # fontsize=45
                      )
        ax.set_title('Response to Outcome',
                     # fontsize=55
                     )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # ax.tick_params(axis='x', which='major', labelsize=35)
        # ax.tick_params(axis='x', which='minor', labelsize=35)
        # plt.xticks(rotation=15)
        # ax.tick_params(axis='y', which='major', labelsize=45)
        # ax.tick_params(axis='y', which='minor', labelsize=45)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
    # ax.legend(fontsize=50)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # plt.tight_layout()
    plt.savefig("figures/cosyne_megaplot_3_empty.jpg", format="jpg")
    plt.savefig("figures/cosyne_megaplot_3_empty.pdf", format="pdf")
    # plt.show(block=False)


def PP_4A(V_foods, V_waters, empty_plot=False, use_baseline=False):
    s1_p1 = 0.9
    s1_p2 = 0.1
    s2_p1 = 0.1
    s2_p2 = 0.9
    satiated_baseline = 0.25
    baseline_baseline = 0.5
    a_food = 0.45
    a_water = 0.40
    deltas_food_received = np.zeros((len(V_foods), 2))
    deltas_water_received = np.zeros((len(V_foods), 2))
    PP_food = 0.3
    PP_water = 0.2
    PP_baseline = 0.5
    for i in range(len(V_foods)):
        # assuming having received food
        V_food = V_foods[i, :]
        V_water = V_waters[i, :]

        delta_food = (s1_p1 * V_food[0]) + (s2_p1 * V_food[1])
        delta_water = (s1_p2 * V_water[0]) + (s2_p2 * V_water[1])
        deltas_food_received[i, :] = np.array([delta_food, -delta_water])

        # assuming having received water!
        delta2_food = (s1_p2 * V_food[0]) + (s2_p2 * V_food[1])
        delta2_water = (s1_p1 * V_water[0]) + (s2_p1 * V_water[1])
        deltas_water_received[i, :] = np.array([-delta_food, delta_water])

    print("DELTAS: ", deltas_food_received)
    total_ds_food = []
    total_ds_water = []
    total_ds_baseline = []
    for i in range(len(V_foods)):
        total_ds_food.append(
            (a_food * deltas_food_received[i, 0]) + (a_water * deltas_food_received[i, 1]))
        total_ds_water.append(
            (a_food * deltas_water_received[i, 0]) + (a_water * deltas_water_received[i, 1]))
        total_ds_baseline.append(
            (deltas_food_received[i, 0] + deltas_food_received[i, 1] + deltas_water_received[i, 0] + deltas_water_received[i, 1])/4)

    print(total_ds_food)
    print(total_ds_water)
    mean_ds_food = np.mean(np.array(total_ds_food)) + satiated_baseline
    mean_ds_water = np.mean(np.array(total_ds_water)) + satiated_baseline
    mean_ds_baseline = np.mean(np.array(total_ds_baseline)) + baseline_baseline
    std_ds_food = np.std(np.array(total_ds_food))
    std_ds_water = np.std(np.array(total_ds_water))
    std_ds_baseline = np.std(np.array(total_ds_baseline))
    print(mean_ds_food)
    print(mean_ds_water)
    print(mean_ds_baseline)
    if use_baseline:
        labels = ["Baseline", "Valued", "Devalued"]
        PP_means = [PP_baseline, PP_food, PP_water]
        sim_means = [mean_ds_baseline, mean_ds_food, mean_ds_water]
    else:
        labels = ["Valued", "Devalued"]
        PP_means = [PP_food, PP_water]
        sim_means = [mean_ds_food, mean_ds_water]
    # if not empty_plot:
    #     sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    #     sns.set_theme('talk', font_scale=1.2)
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/1.5, PP_means, width,
                  label='Experiment')
    if use_baseline:
        bar2 = ax.bar(x + width/1.5, sim_means, width,
                      yerr=[std_ds_baseline, std_ds_food, std_ds_water], label='Simulation')
    else:
        bar2 = ax.bar(x + width/1.5, sim_means, width,
                      yerr=[std_ds_food, std_ds_water], label='Simulation')
    if not empty_plot:
        ax.set_ylabel(
            'Relative Dopamine',
            # fontsize=50
        )
        ax.set_title(
            'Response to Cue at First Trial',
            # fontsize=60
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # ax.tick_params(axis='x', which='major', labelsize=45)
        # ax.tick_params(axis='x', which='minor', labelsize=45)
        # ax.tick_params(axis='y', which='major', labelsize=45)
        # ax.tick_params(axis='y', which='minor', labelsize=45)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
    # ax.legend(fontsize=50)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # fig.tight_layout()
    # plt.tight_layout()
    if use_baseline:
        plt.savefig("figures/proper_PP_4A_4_empty.jpg", format="jpg")
    else:
        plt.savefig(
            "figures/proper_PP_4A_4_empty_no_baseline.jpg", format="jpg")
        plt.savefig(
            "figures/proper_PP_4A_4_empty_no_baseline.pdf", format="pdf")
    # plt.show(block=False)


def Vs_N_experiments(N_runs, N_steps=500, lr=0.1):
    V_foods = np.zeros((N_runs, 2))
    V_waters = np.zeros((N_runs, 2))
    for i in range(N_runs):
        V_food, V_water = train_agent_papageorgiou_exp(N_steps=N_steps, lr=lr)
        V_foods[i, :] = V_food
        V_waters[i, :] = V_water
    np.save("data/PP_V_foods.npy", V_foods)
    np.save("data/PP_V_waters.npy", V_waters)
    return V_foods, V_waters


if __name__ == '__main__':
    REGENERATE_DATA = False
    # plot_pp_1C()
    # plot_rb_1C()
    # plot_pp_2AB()
    # plot_rb_2AB()
    # plot_rb_1D()
    # plot_pp_1D()
    # train_agent_papageorgiou_exp()
    #V_foods = np.load("V_foods.npy")
    #V_waters = np.load("V_waters.npy")
    if REGENERATE_DATA:
        V_foods, V_waters = Vs_N_experiments(20)
    else:
        V_foods = np.load("data/PP_V_foods.npy")
        V_waters = np.load("data/PP_V_waters.npy")
    #pp_2A(V_foods, V_waters)

    #V_foods, V_waters = Vs_N_experiments(20)
    #PP_2C(V_foods, V_waters,average_conditions = True)

    #V_foods, V_waters = Vs_N_experiments(20)
    #PP_2D(V_foods, V_waters)

    PP_4A(V_foods, V_waters, empty_plot=False, use_baseline=False)
    cosyne_combined_megaplot(V_foods, V_waters, empty_plot=False)
