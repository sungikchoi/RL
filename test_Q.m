function[sumrewards]=test_Q(init_state, n_states, n_actions, testtime, alpha, gamma, reward, terminal, new_state,Q)
n_trials=zeros(testtime,1);
T=1e3;
testrewards=zeros(testtime,1);
for k=1:testtime
    s=init_state;
    rewards=zeros(1,T);
    for j=1:T
        [~,a]=max(Q(s,:));
        %a=randsample(n_actions,1,true,pi(s,:)); %select action
        sn=new_state(s,a);
        rewards(1,j)=reward(s,a);
        if terminal(sn)
            n_trials(k)=j;
            break;
        end
        s=sn;
    end
    testrewards(k)=sum(rewards);
end
sumrewards=sum(testrewards);
end