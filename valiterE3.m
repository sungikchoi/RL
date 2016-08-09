function[pi]=valiterE3(n_states,n_actions,horizontime,reward,newstate,gamma)
V=zeros(horizontime+1,n_states);
pi=zeros(horizontime,n_states);
for a=1:horizontime
    t=horizontime+1-a;
    for i=1:n_states
        actionarray=zeros(n_actions,1);
        for j=1:n_actions
            if(newstate(i,j)~=0)
                actionarray(j)=gamma*V(t+1,newstate(i,j));
            end
        end
        [maxvalue,maxaction]=max(actionarray);
        V(t,i)=reward(i)+maxvalue;
        maxcount=0;
        indexarray=zeros(n_actions,1);
        for check=1:n_actions
            if(actionarray(check)==maxvalue)
                maxcount=maxcount+1;
                indexarray(maxcount)=check;
            end
        end
        temp1=randi(maxcount);
        pi(t,i)=indexarray(temp1);
    end
end
end