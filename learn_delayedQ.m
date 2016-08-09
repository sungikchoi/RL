function[Q,n_trials]=learn_delayedQ(init_state, n_states, n_actions, n_episodes, alpha, gamma, reward, terminal, new_state)
%constant
epsilon1= 0.01;
kappa=1/(1-gamma)/epsilon1;
delta=0.1;
m=2;
Q=zeros(n_states,n_actions);
U=zeros(n_states,n_actions);
l=zeros(n_states,n_actions);
t=zeros(n_states,n_actions);
LEARN=zeros(n_states,n_actions);
n_trials=zeros(n_episodes,1);
for stateiter=1:n_states
    for actioniter=1:n_actions
        if(new_state(stateiter,actioniter)~=0)
            Q(stateiter,actioniter)=1;
            LEARN(stateiter,actioniter)=1;
        end
    end
end
tstar=0;
for k=1:n_episodes
    t=zeros(n_states,n_actions);
    s=init_state;
    for j=1:1e3
        [maxvalue,~]=max(Q(s,:));
        maxcount=0;
        indexarray=zeros(n_actions,1);
        for i=1:n_actions
            if(Q(s,i)==maxvalue)
                maxcount=maxcount+1;
                indexarray(maxcount)=i;
            end
        end
        temp1=randi(maxcount);
        a=indexarray(temp1);
        rewardr=reward(s,a);
        sn=new_state(s,a);
        if LEARN(s,a)==1
            [maxvaluenext,~]=max(Q(sn,:));
            U(s,a)=U(s,a)+rewardr+gamma*maxvaluenext;
            l(s,a)=l(s,a)+1;
            if l(s,a)==m
                if (Q(s,a)-U(s,a)/m) >= 2*epsilon1
                    Q(s,a)=U(s,a)/m+epsilon1;
                    tstar=k;
                else
                    if t(s,a)>=tstar
                        LEARN(s,a)=0;
                    end
                end
                t(s,a)=j; U(s,a)=0; l(s,a)=0;
            end
        else
            if t(s,a)<tstar
                LEARN(s,a)=1;
            end
        end
        if terminal(sn)
            n_trials(k)=j;
            break;
        end
        s=sn;
    end
end
end