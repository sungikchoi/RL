% state action  next  reward  terminal 
%   1      A      2      0      Yes
%   1      B      3      0      No
%   3      A      4      1      Yes
%   3      B      5      0      Yes

n_states=5;       % number of states
n_actions=2;      % number of actions
n_episodes=10000;   % number of episodes to run
alpha=0.2;        % learning rate
gamma=0.9;        % discount factor
reward=[0 0;0 0;1 0;0 0;0 0];      % reward for each (state,action)
terminal=[0;1;0;1;1];              % 1 if terminal state, 0 otherwise
new_state=[2 3;0 0;4 5;0 0;0 0];   % new_state

init_state=1;     % initial state

psi=zeros(n_states,n_actions,n_states); % z axis implies the probabilistic vector
psi(1,1,2)=1;
psi(1,2,3)=1;
psi(3,1,4)=1;
psi(3,2,5)=1;


% Q learning
%[Q,n_trials,rewards]=learn_Q(init_state, n_states, n_actions, n_episodes, alpha, gamma, reward, terminal, new_state);
%[pi,n_trials]=learn_softmaxREINFORCE(init_state, n_states, n_actions, n_episodes, alpha, gamma, reward, terminal, new_state,psi);
[pi,n_trials]=learn_softmaxAC(init_state, n_states, n_actions, n_episodes, alpha,gamma, gamma, reward, terminal, new_state,psi);
pi
