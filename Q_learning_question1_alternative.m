n_states=10;
n_actions=2;
n_episodes=5000;
alpha=0.1;
gamma=0.9;

reward=zeros(n_actions,n_states);
reward(1,2)=0.4;
reward(2,9)=1;
terminal=[1 0 0 0 0 0 0 0 0 1];

new_state=zeros(n_actions,n_states);


for j=2:n_states-1
    new_state(1,j)=j-1;
    new_state(2,j)=j+1;
end
reward=transpose(reward);
terminal=transpose(terminal);
new_state=transpose(new_state);
init_state=2;
% Q learning
%[Q,n_trials,rewards]=learn_Q(init_state, n_states, n_actions, n_episodes, alpha, gamma, reward, terminal, new_state);

%sum(n_trials)
%Q

%softmax policy learning

psi=zeros(n_states,n_actions,n_states); % z axis implies the probabilistic vector
epsilon=0.05;
psi=psi+epsilon*ones(n_states,n_actions,n_states);
for i=2:n_states-1
    psi(i,1,i-1)=1-epsilon;
    psi(i,2,i+1)=1-epsilon;
end

Rewards=zeros(10,1);
Numk=zeros(10,1);
for k=1:10
    for s=1:1000
       n_episodes=20*k;
       Numk(k)=20*k;
       [Q,n_trials,~]=learn_Q(init_state, n_states, n_actions,n_episodes,alpha, gamma, reward, terminal, new_state);
       answer=test_Q(init_state,n_states,n_actions,1,alpha,gamma,reward,terminal,new_state,Q);
       Rewards(k)=Rewards(k)+answer;
    end
end
plot(Numk,Rewards/1000,'-*')