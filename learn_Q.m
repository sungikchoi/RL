function [Q,n_trials,rewards]=learn_Q(init_state, n_states, n_actions, n_episodes, alpha, gamma, reward, terminal, new_state)
Q=zeros(n_states,n_actions);
rand0=1;
rand1=0;
n_trials=zeros(n_episodes,1);
rewards=zeros(n_episodes,1);
for k=1:n_episodes
    randomness=(rand1-rand0)*(k/n_episodes)^1+rand0;
    s=init_state;
    for j=1:1e9
        if rand(1)<randomness
            a=randi(n_actions);    % random
        else
            [maxvalue,a]=max(Q(s,:));
            maxcount=0;
            indexarray=zeros(n_actions,1);
            for check=1:n_actions
                if(Q(s,check)==maxvalue)
                    maxcount=maxcount+1;
                    indexarray(maxcount)=check;
                    
                end
            end   
            temp1=randi(maxcount);
            a=indexarray(temp1);
        end
        sn=new_state(s,a);
        r=reward(s,a);
        rewards(k)=rewards(k)+r;
        Q(s,a)=(1-alpha)*Q(s,a)+alpha*(r+gamma*max(Q(sn,:)));
        if terminal(sn)
            n_trials(k)=j;
            break;
        end
        s=sn;
    end
end
end
