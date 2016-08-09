n_states=10;
n_actions=2;
n_episodes=1000;
alpha=0.2;
gamma=0.9;
max_steps=1e9;

reward=zeros(n_actions,n_states);
reward(2,9)=1;
reward(1,2)=0.3;
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
oldrewards=zeros(10,1);
newrewards=zeros(10,1);
Numk=zeros(10,1);
n_episodes=100;
delayedreward=zeros(90,1);
e3reward=zeros(90,1);
backtrackreward=zeros(990,1);
Qreward=zeros(990,1);
for i=1:1000;
    %[Q,n_trials]=learn_delayedQ(init_state, n_states, n_actions,100,alpha, gamma, reward, terminal, new_state);
    %delayedreward=delayedreward+n_trials(11:100)/10000;
    %[pi,n_trials]=learn_ExploitExploreorExploit(init_state, n_states, n_actions, 100, alpha, gamma, reward, terminal, new_state);
    %e3reward=e3reward+n_trials(11:100)/10000;
    [Q,n_trials,rewards]=learn_Q_with_backtracking(init_state, n_states, n_actions, 1000, max_steps, alpha, gamma, reward, terminal, new_state);
    backtrackreward=backtrackreward+n_trials(11:1000)/1000;
    [Q,n_trials,rewards]=learn_Q(init_state, n_states, n_actions, 1000, alpha, gamma, reward, terminal, new_state);
    Qreward=Qreward+n_trials(11:1000)/1000;
end
Numk=zeros(990,1);
for i=1:990
    Numk(i)=i+10;
end
hold off;
%plot(Numk,delayedreward,'-*');
hold on;
%plot(Numk,e3reward,'--o');
plot(Numk,backtrackreward,'-O');
plot(Numk,Qreward,':+');
%legend('Delayed Q learning', 'E3 algorithm', 'Backtracking','Qlearning');
legend('Backtracking','Qlearning');
    

%[Q,n_trials,rewards]=learn_Q_with_backtracking(init_state, n_states, n_actions, 50, max_steps, alpha, gamma, reward, terminal, new_state)
%for k=1:10
%    for s=1:1000
%       n_episodes=100*k;
%       Numk(k)=100*k;
%       [Q,n_trials,rewards]=learn_Q(init_state, n_states, n_actions,n_episodes,alpha, gamma, reward, terminal, new_state);
%       answer=test_Q(init_state,n_states,n_actions,1,alpha,gamma,reward,terminal,new_state,Q);
%       newrewards(k)=newrewards(k)+answer;
%       oldrewards(k)=oldrewards(k)+sum(rewards)/(n_episodes);
%    end
%end
%plot(Numk,oldrewards/1000,'-*');
%hold on;
%plot(Numk,newrewards/1000,'-o')
%legend('average rewards during training', 'Rewards on test');
%hold off;



