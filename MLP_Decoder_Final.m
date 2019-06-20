% clear;
% close all;
clc;
tic

%% Initial Parameters

number_of_codeword = 10000;

H=[1 1 1 1 0 0 0 0 0 0       % LDPC H Matrix
   1 0 0 0 1 1 1 0 0 0
   0 1 0 0 1 0 0 1 1 0
   0 0 1 0 0 1 0 1 0 1
   0 0 0 1 0 0 1 0 1 1];

G=[1 1 0 0 1 0 0 0 0 0
   1 0 1 0 0 1 0 0 0 0
   1 0 0 1 0 0 1 0 0 0
   0 1 1 0 0 0 0 1 0 0
   0 1 0 1 0 0 0 0 1 0
   0 0 1 1 0 0 0 0 0 1];

Code_length = size(G,2);
K = size(G,1);

p=length(find(H(1,:)));    %number of 1 in each row
Check_Node_Threshold = .2;
eta = .008;                  %train rate
max_iteratin = 200;
temp=zeros(size(H,1),p);
SNR_dB_vector = 0:11;



%% Generating Data

 data  = randi([0 1], 1, K*number_of_codeword);
 Encoded_data=[];
 

 %% LDPC Encoder
 
for l=1:6:length(data)
  temp_encoded_data = data(l:l+5)*G;
  for k=1:length(temp_encoded_data)
      if mod(temp_encoded_data(k),2)==0
          temp_encoded_data(k)=0;
      else
          temp_encoded_data(k)=1;
      end
  end
  Encoded_data = [Encoded_data temp_encoded_data];    %#ok
end


%% BPSK Modulation 

Modulated_Encoded_data = [];

for i=1:length(Encoded_data)                                    %Random bit generation
    if Encoded_data(i) == 0
       Modulated_Encoded_data(i) = -1;   %#ok
    else
       Modulated_Encoded_data(i) = 1;    %#ok
    end
end  


%% Adding AWGN Noise

for s = 1:length(SNR_dB_vector)
    
%     noise_varianc = 1 / SNR(s);
%     noise_varianc_dB = 10 * log10( noise_varianc);
%     noise = wgn(1,length(Modulated_Encoded_data),noise_varianc_dB,'complex');
      Received_data = awgn(Modulated_Encoded_data,SNR_dB_vector(s));
%     Received_data = Modulated_Encoded_data + noise;

    for z=1:length(Encoded_data)                                   
        if Received_data(z) >= 0   
            Received_data(z)=1;
        else
            Received_data(z)=0;
        end
    end
    

    %% MLP Decoder
    
    r_hat = Received_data;
    Correct_Data =[];

    for l=1:10:length(Received_data)
        temp_encoded_data = Received_data(l:l+9);
        r_hat_temp = r_hat(l:l+9);

        for k=1:p
            for m=1:size(H,1)              
                c(m,:)=find(H(m,:));       %#ok
                temp(m,k)=temp_encoded_data(c(m,k));       %findig bits who has role in calculating  each check nodes
            end
        end

        for k=1:max_iteratin
            temp(1,1)=r_hat_temp(1); 
            temp(2,1)=r_hat_temp(1);
            temp(1,2)=r_hat_temp(2);
            temp(3,1)=r_hat_temp(2);
            temp(1,3)=r_hat_temp(3);
            temp(4,1)=r_hat_temp(3);
            temp(1,4)=r_hat_temp(4);
            temp(5,1)=r_hat_temp(4);
            temp(2,2)=r_hat_temp(5);
            temp(3,2)=r_hat_temp(5);
            temp(2,3)=r_hat_temp(6);
            temp(4,2)=r_hat_temp(6);
            temp(2,4)=r_hat_temp(7);
            temp(5,2)=r_hat_temp(7);
            temp(3,3)=r_hat_temp(8);
            temp(4,3)=r_hat_temp(8);
            temp(3,4)=r_hat_temp(9);
            temp(5,3)=r_hat_temp(9);
            temp(4,4)=r_hat_temp(10);
            temp(5,4)=r_hat_temp(10);

            for i=1:size(temp,1)                                           %size(temp,1)= number of check nodes
                t=temp(i,1)*(1-temp(i,2))+temp(i,2)*(1-temp(i,1));         %result: XOR of 2 input
                tt=t*(1-temp(i,3))+temp(i,3)*(1-t);                        %result: XOR of 3 input
                O(i)=tt*(1-temp(i,4))+temp(i,4)*(1-tt);     %#ok           %result: XOR of 4 input    
            end
            
%             sse_output(k)= 0.5*sum(O.^2);                                  %#ok
%             if sse_output(k) < Check_Node_Threshold
%                 break
%             end
            mse_output(k)=mse(O);                                  %#ok
            if mse_output(k) < Check_Node_Threshold
                break
            end

    % Training Mode

            r_hat_temp(1)=r_hat_temp(1)-eta*(O(1)*(1-2*r_hat_temp(2))*(1-2*r_hat_temp(3))*(1-2*r_hat_temp(4))+O(2)*(1-2*r_hat_temp(5))*(1-2*r_hat_temp(6))*(1-2*r_hat_temp(7)));
            r_hat_temp(2)=r_hat_temp(2)-eta*(O(1)*(1-2*r_hat_temp(1))*(1-2*r_hat_temp(3))*(1-2*r_hat_temp(4))+O(3)*(1-2*r_hat_temp(5))*(1-2*r_hat_temp(8))*(1-2*r_hat_temp(9)));
            r_hat_temp(3)=r_hat_temp(3)-eta*(O(1)*(1-2*r_hat_temp(1))*(1-2*r_hat_temp(2))*(1-2*r_hat_temp(4))+O(4)*(1-2*r_hat_temp(6))*(1-2*r_hat_temp(8))*(1-2*r_hat_temp(10)));
            r_hat_temp(4)=r_hat_temp(4)-eta*(O(1)*(1-2*r_hat_temp(1))*(1-2*r_hat_temp(2))*(1-2*r_hat_temp(3))+O(5)*(1-2*r_hat_temp(7))*(1-2*r_hat_temp(9))*(1-2*r_hat_temp(10)));
            r_hat_temp(5)=r_hat_temp(5)-eta*(O(2)*(1-2*r_hat_temp(1))*(1-2*r_hat_temp(6))*(1-2*r_hat_temp(7))+O(3)*(1-2*r_hat_temp(2))*(1-2*r_hat_temp(8))*(1-2*r_hat_temp(9)));
            r_hat_temp(6)=r_hat_temp(6)-eta*(O(2)*(1-2*r_hat_temp(1))*(1-2*r_hat_temp(5))*(1-2*r_hat_temp(7))+O(4)*(1-2*r_hat_temp(3))*(1-2*r_hat_temp(8))*(1-2*r_hat_temp(10)));
            r_hat_temp(7)=r_hat_temp(7)-eta*(O(2)*(1-2*r_hat_temp(1))*(1-2*r_hat_temp(5))*(1-2*r_hat_temp(6))+O(5)*(1-2*r_hat_temp(4))*(1-2*r_hat_temp(9))*(1-2*r_hat_temp(10)));
            r_hat_temp(8)=r_hat_temp(8)-eta*(O(3)*(1-2*r_hat_temp(2))*(1-2*r_hat_temp(5))*(1-2*r_hat_temp(9))+O(4)*(1-2*r_hat_temp(3))*(1-2*r_hat_temp(6))*(1-2*r_hat_temp(10)));
            r_hat_temp(9)=r_hat_temp(9)-eta*(O(3)*(1-2*r_hat_temp(2))*(1-2*r_hat_temp(5))*(1-2*r_hat_temp(8))+O(5)*(1-2*r_hat_temp(4))*(1-2*r_hat_temp(7))*(1-2*r_hat_temp(10)));
          r_hat_temp(10)=r_hat_temp(10)-eta*(O(4)*(1-2*r_hat_temp(3))*(1-2*r_hat_temp(6))*(1-2*r_hat_temp(8))+O(5)*(1-2*r_hat_temp(4))*(1-2*r_hat_temp(7))*(1-2*r_hat_temp(9)));

        end


        for i=1:Code_length
            if abs(r_hat_temp(i)-0) > abs(r_hat_temp(i)-1)
                r_correct(i)=1;                %#ok
            else
                r_correct(i)=0;                %#ok 
            end
        end

        Correct_Data(1,l:l+9) = r_correct;

    end
    
    decoded_message = [];
    for i = 1:Code_length:length(Correct_Data)
        temp_message = Correct_Data(i+4:i+9);
        decoded_message = [decoded_message temp_message];                  %#ok
    end


    %% Performance of Corrected data

    Error_message_stream = abs(data - decoded_message);
    BER_corrected(s) = sum(Error_message_stream)/length(data);             %#ok
    
    %% Decoding without Correcting

%     channel_error(s) = sum(abs(Received_data-Encoded_data));               %#ok                         
%     % BER_channel = channel_error/length(Encoded_data);
%     decoded_uncorrect = [];
%         for ii = 1:Code_length:length(Received_data)
%             temp_uncorrect = Received_data(ii+4:ii+9);
%             decoded_uncorrect = [decoded_uncorrect temp_uncorrect];        %#ok
%         end
%     Error_uncoded_stream = abs(data - decoded_uncorrect);
%     BER_uncorrected(s) = sum(Error_uncoded_stream)/length(data);           %#ok 

end
toc
semilogy(SNR_dB_vector,BER_corrected)

