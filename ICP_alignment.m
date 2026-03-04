in_als_file = "C:/Users/digit/Downloads/Examensarbete/Data/las_polygon/240829_ALS_Matrice300_Svb_clipped.las";
in_tls_file = "C:/Users/digit/Downloads/Examensarbete/Data/radarTower001_clipped.las";
out_file    = "C:/Users/digit/Downloads/Examensarbete/Results/TLS_aligned.las";
ALS = lasFileReader(in_als_file);
TLS = lasFileReader(in_tls_file);
alscloud = readPointCloud(ALS);
tlscloud = readPointCloud(TLS);
fprintf("Successfully read files")
tlscloud_sub = pcdownsample(tlscloud,'gridAverage',0.1);
fprintf("Successfully downsampled")
tform = pcregistericp(tlscloud_sub,alscloud);
disp(['Final RMS Error: ', num2str(tform.RMSE)]);
fprintf("Successfully made transform")
ptCloudTformed = pctransform(tlscloud,tform);
fprintf("Successfully transformed")

lasWriter = lasFileWriter(out_file);
writePointCloud(lasWriter,ptCloudTformed);

fprintf('Successfully saved transformed LAS to: %s\n', out_file);