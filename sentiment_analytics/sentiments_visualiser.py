import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

if os.path.exists("user_dist.pkl"):
    with open('user_dist.pkl', 'rb') as f:
        user_emo_dist = pkl.load(f)
else:
    print("Error: No user_dist.pkl found")

id2emo = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

def plot_mu_per_user():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes = axes.flatten()

    i = 0
    for user, values in user_emo_dist.items():
        if user == '__GrOuP__':
            continue
        ax = axes[i]
        ax.bar(list(id2emo.values()), values['mu'])
        ax.set_title(f"Mean Emotion Distribution for {user}")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotion")
        i += 1

    plt.show()

def plot_hist_per_user():
    for user, values in user_emo_dist.items():
        if user == '__GrOuP__':
            continue
        hist = np.array(values['hist'])
        plt.figure()
        for i, emo in id2emo.items():
            plt.plot(hist[:, i], label=emo)
        plt.title(f"Emotion History for {user}")
        plt.xlabel("Message Index")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

def plot_coeffvar_per_user():
    for user, values in user_emo_dist.items():
        if user == '__GrOuP__':
            continue
        plt.figure()
        plt.bar(list(id2emo.values()), values['coeffvar'])
        plt.title(f"Coefficient of Variation for {user}")
        plt.ylabel("Coeff. of Variation")
        plt.xlabel("Emotion")
        plt.show()

def user_volatility():
    for user, values in user_emo_dist.items():
        if user == '__GrOuP__':
            continue
        print(f"{user} is volatile: {values['volatile']}")

if __name__ == "__main__":
    plot_mu_per_user()
    # plot_hist_per_user()
    # plot_coeffvar_per_user()
    # user_volatility()

