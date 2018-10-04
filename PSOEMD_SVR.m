function PSOEMD_SVR(target)
%Len=size(target,1);
% len1=round(Len*0.8);
imf=emd(target,1);
TT=target-imf(end,:)'+mean(imf(end,:));
len1=100;

nVar=len1;            % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix
lamda=1/2;
VarMin=-1/lamda;         % Lower Bound of Variables
VarMax= 1/lamda;         % Upper Bound of Variables


%% PSO Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=20;        % Population Size (Swarm Size)

% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient


% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(TT,particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(TT,particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
%     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

BestSol = GlobalBest;
predictdata=predict_psosvr(target, BestSol.Position,len1,50);
Len=len1+50;
mape=100/(Len-len1)*sum(abs(target(len1+1:Len)-predictdata)./target(len1+1:Len));
rmse=1/(Len-len1)*sqrt((target(len1+1:Len)-predictdata)'*(target(len1+1:Len)-predictdata));
%% Results
figure;
plot(target(1:Len),'b');
hold on; plot(len1+1:Len,predictdata,'r');
title(['PSOEMD_SVR model(mape=', num2str(mape),', rmse=',num2str(rmse),')']);
legend( 'PSO_SVR Result','EMD Result');
end
function z=CostFunction(adata, coe)
coe=coe';
sigma=0.75;
z=0;
mm=100;
for i=1:mm
    for j=1:mm
        s1=coe(i)*coe(j);
        s2=exp(-(adata(i:i+9,1)-adata(j:j+9,1))'*(adata(i:i+9,1)-adata(j:j+9,1))/(2*sigma));
        z=z+0.5*s1*s2;
    end
     s3=coe(i)*adata(i+10)-sign(coe(i));
     z=z+s3+1E+5*abs(coe(i));
end
end
function z=predict_psosvr(adata, coe,len1,num)
coe=coe';
sigma=0.75;
z=zeros(num,1);
mm=100;
for k=1:num
    z(k)=0;
    for i=1:mm
        x=adata(len1+k-10:len1+k-1,1);
        y=adata(i:i+9,1);
        s2=exp(-(x-y)'*(x-y)/(2*sigma));
        z(k)=z(k)+coe(i)*s2;
    end
    b=mean(adata(len1+k-10:len1+k-1,1))-z(k)/mm;
    z(k)=z(k)+b;
end
end