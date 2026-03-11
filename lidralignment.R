#install.packages(c('lidR', 'lasR', 'lidRalignment'), repos = 'https://r-lidar.r-universe.dev')
library(lidR)
library(lidRalignment)
options(lidR.progress = FALSE)

fref = "C:/Users/digit/Downloads/Examensarbete/Results/trees_lidr/lidr_segmentation__instance_103__n5695__score_103.las"
fmov = "C:/Users/digit/Downloads/Examensarbete/Results/trees_TLS_labeled_from_ALS_lidr_260304_clipped/TLS_labeled_from_ALS_lidr_260304_clipped__instance_103__n4076.las"

# Setup the pipeline. It is important to tell the object
# what we are aligning in order to perform or not
# the last extra fine alignment
alignment = AlignmentScene$new(fref, fmov)
alignment$set_ref_is_ground_based(FALSE)
alignment$set_mov_is_ground_based(TRUE)
alignment$set_radius(20)

# Run the alignment pipeline
alignment$align()

# Visualize the different level of alignment
alignment$plot("raw")
alignment$plot("coarse")
alignment$plot("fine")
alignment$extra_fine_align()
alignment$plot("extra", compare_to = "fine")

# Get the final transformation matrix to register the entire point cloud.
M = alignment$get_registration_matrix()

crs = sf::st_crs(readLASheader(fref))
ofile = transform_las(fmov, M, crs)

writeLAS(ofile, file = "C:/Users/digit/Downloads/Examensarbete/Results/instance103_aligned_lidrseg_TLS_clipped.las" )
