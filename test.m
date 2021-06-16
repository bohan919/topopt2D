clear
%%% top88(nelx,nely,volfrac,penal,rmin,ft)
i = 1;
% for nely = 60:150
    nelx = 200;
    nely = 100;
    
%     fig1 = figure(1);
%     tic
%     top88(nelx,nely,0.4, 2, 2, 2);
%     toc
%     saveas(fig1, nelx+"_"+nely+"_Raw.png")
    disp('AM')
    fig2 = figure(2);
    tic
    top88_AMfilter(nelx,nely,0.4, 2, 2, 2);
    toc
%     saveas(fig2, nelx+"_"+nely+"_AMFull.png")
    
%     close all
% end
% top88(90,30,0.4, 5, 2, 3);
% top88_AMfilter(200,100,0.4, 5, 2, 3);
% top88(40,20,0.4, 2, 2, 2);
% top88_AMfilter(40,20,0.4, 2, 2, 2);
% top3d(30,40,5,0.4,5,2);

% x = [ 1 0 1 1; 1 1 0 1; 1 1 0 1]
% dc = [0.2 0.2 0.2 0.2; 0.3 0.3 0.3 0.3; 0.4 0.4 0.4 0.4]
% dv = dc -0.1
% baseplate = 'N'
% varIn = {dc, dv}
% 
% %INTERNAL SETTINGS
% P = 40; ep = 1e-4; xi_0 = 0.5; % parameters for smooth max/min functions
% 
% %INPUT CHECKS
% 
% nRot=find(upper(baseplate)=='SWNE')-1;
% 
% 
% %ORIENTATION
% x=rot90(x,nRot);
% xi=zeros(size(x));
% for s=1:2
%     varIn{s}=rot90(varIn{s},nRot);    
% end
% [nely,nelx]=size(x); 
% 
% %AM FILTER =====================
% Ns=3;
% Q=P+log(Ns)/log(xi_0); 
% SHIFT = 100*realmin^(1/P); % small shift to prevent division by 0
% BACKSHIFT = 0.95*Ns^(1/Q)*SHIFT^(P/Q);
% Xi=zeros(size(x)); keep=zeros(size(x)); sq=zeros(size(x));
% % baseline: identity
% xi(nely,:)=x(nely,:); % copy base row as-is
% for i=(nely-1):-1:1
%     % compute maxima of current base row
%     cbr = [0, xi(i+1,:), 0] + SHIFT; % pad with zeros
%     keep(i,:) = (cbr(1:nelx).^P + cbr(2:(nelx+1)).^P + cbr(3:end).^P);
%     Xi(i,:) = keep(i,:).^(1/Q) - BACKSHIFT;
%     sq(i,:) = sqrt((x(i,:)-Xi(i,:)).^2 + ep);
%     % set row above to supported value using smooth minimum:
%     xi(i,:) = 0.5*((x(i,:)+Xi(i,:)) - sq(i,:) + sqrt(ep));
% end
% %SENSITIVITIES
% 
%     dfxi=varIn; dfx=varIn; 
%     lambda=zeros(2,nelx); 
%     % from top to base layer:
%     for i=1:nely-1
%         % smin sensitivity terms
%         dsmindx  = .5*(1-(x(i,:)-Xi(i,:))./sq(i,:));
%         %dsmindXi = .5*(1+(x(i,:)-Xi(i,:))./sq(i,:)); 
%         dsmindXi = 1-dsmindx; 
%         % smax sensitivity terms
%         cbr = [0, xi(i+1,:), 0] + SHIFT; % pad with zeros
%         dmx = zeros(Ns,nelx);
%         for j=1:Ns
%             dmx(j,:) = (P/Q)*keep(i,:).^(1/Q-1).*cbr((1:nelx)+(j-1)).^(P-1);
%         end        
%         % rearrange data for quick multiplication:
%         qj=repmat([-1 0 1]',nelx,1);
%         qi=repmat(1:nelx,3,1); qi=qi(:);
%         qj=qj+qi; qs=dmx(:);
%         dsmaxdxi=sparse(qi(2:end-1),qj(2:end-1),qs(2:end-1)); 
%         for k=1:2
%             dfx{k}(i,:) = dsmindx.*(dfxi{k}(i,:)+lambda(k,:));
%             lambda(k,:)= ((dfxi{k}(i,:)+lambda(k,:)).*dsmindXi)*dsmaxdxi;
%         end
%     end
%     % base layer:
%     i=nely;
%     for k=1:2
%         dfx{k}(i,:) = dfxi{k}(i,:)+lambda(k,:);
%     end
% 
% 
% %ORIENTATION
% xi=rot90(xi,-nRot);
% for s=1:2
%     varOut{s}=rot90(dfx{s},-nRot) 
% end