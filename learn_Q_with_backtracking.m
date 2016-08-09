function [Q,n_trials,rewards]=learn_Q_with_backtracking(init_state, n_states, n_actions, n_episodes, max_steps, alpha, gamma, reward, terminal, new_state)
Q=zeros(n_states,n_actions);
rand0=1;
rand1=0;
n_trials=zeros(n_episodes,1);
rewards=zeros(n_episodes,1);
mem_size=1000000;
replay_memory=zeros(mem_size,4);
r_size=0;
for k=1:n_episodes
    randomness=(rand1-rand0)*((k-1)/(n_episodes-1))^1+rand0;
    s=init_state;
    for j=1:max_steps
        if rand(1)<randomness
            a=randi(n_actions);    % random
        else
            mq=max(Q(s,:));
            if mq>0
                [mx,a]=max(Q(s,:)+1e-10*mq);
            else
                a=randi(n_actions);
            end
        end
        sn=new_state(s,a);
        r=reward(s,a);
        r_size=r_size+1;
        replay_memory(r_size,:)=[s a r sn];
        rewards(k)=rewards(k)+r;
        Q(s,a)=(1-alpha)*Q(s,a)+alpha*(r+gamma*max(Q(sn,:)));
        if terminal(sn)
            n_trials(k)=j;
            break;
        end
        s=sn;
    end
    for j=r_size:-1:1
        s=replay_memory(j,1);
        a=replay_memory(j,2);
        r=replay_memory(j,3);
        sn=replay_memory(j,4);
        Q(s,a)=(1-alpha)*Q(s,a)+alpha*(r+gamma*max(Q(sn,:)));
    end
end
