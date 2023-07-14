import numpy as np

class HMM:
    def __init__(self, states_count, effects_count) -> None:
        self.status_count = states_count
        self.effects_count = effects_count
        self.evidences = []
        self.transition_matrix = []
        self.emission_matrix = []
        self.initial_status_probs = []
    
    def set_inputs(self, tm, em, evid, init):
        self.transition_matrix = tm
        self.emission_matrix = em
        self.evidences = np.array(evid)
        self.initial_status_probs = np.array(init)

    def forward(self, t):
        if t<1:
            return self.initial_status_probs
        else:
            prev_prob = self.forward(t-1)
            state_t_probs = np.matmul(prev_prob, self.transition_matrix)
            effect_t_prob = np.matmul(state_t_probs, np.diag(self.emission_matrix[:, int(self.evidences[t-1])]))
            return effect_t_prob

def main():
    day_count = int(input())
    status_count = int(input())
    color_count = int(input())
    hmm = HMM(status_count, color_count)
    colors_in_days = [(int(x)-1) for x in input().strip().split(" ")]
    initial_status_probs = [float(x) for x in input().strip().split(" ")]
    tm = np.zeros((status_count, status_count))
    for i in range(status_count):
        tm[i] = np.array([float(x) for x in input().strip().split(' ')])
    em = np.zeros((status_count, color_count))
    for i in range(status_count):
        em[i] = np.array([float(x) for x in input().strip().split(' ')])
    hmm.set_inputs(tm, em, colors_in_days, initial_status_probs)
    next_probs = np.matmul(hmm.forward(day_count), hmm.transition_matrix)
    next_effect_probs = np.matmul(next_probs, hmm.emission_matrix)
    chosen_color = np.argmax(next_effect_probs)
    normalized_probs = next_effect_probs / sum(next_effect_probs)
    print( chosen_color+1, '{:.2f}'.format(normalized_probs[chosen_color]))

main()