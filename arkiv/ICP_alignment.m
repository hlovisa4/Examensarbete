in_als_file = "C:/Users/digit/Downloads/Examensarbete/Results/inst_39_ALS.las";
in_tls_file = "C:/Users/digit/Downloads/Examensarbete//Results/inst_39_TLS.las";
out_file    = "C:/Users/digit/Downloads/Examensarbete/Results/inst_39_aligned.las";
ALS = lasFileReader(in_als_file);
TLS = lasFileReader(in_tls_file);
alscloud = readPointCloud(ALS);
tlscloud = readPointCloud(TLS);
fprintf("Successfully read files")
tlscloud_sub = pcdownsample(tlscloud,'gridAverage',0.1);
fprintf("Successfully downsampled")
[tform, movingReg] = pcregistericp(tlscloud_sub,alscloud);
%disp(['Final RMS Error: ', num2str(tform.RMSE)]);
fprintf("Successfully made transform")
ptCloudTformed = pctransform(tlscloud,tform);
fprintf("Successfully transformed")

save("C:/Users/digit/Downloads/Examensarbete/Results/icp_results_11.mat", "tform", "ptCloudTformed", "tlscloud_sub");
clear TLS tlscloud tlscloud_sub

lasWriter = lasFileWriter(out_file);
writePointCloud(lasWriter,movingReg);
%writePointCloud(lasWriter,ptCloudTformed);

fprintf('Successfully saved transformed LAS to: %s\n', out_file);