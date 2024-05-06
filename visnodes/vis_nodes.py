import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt

def compute_new_node_attribute(node_data, centroid_data, similarity_type='cosine'):
    # Compute the cosine similarity for each vector in cluster_data
    new_node_attribute = 1 - (np.dot(node_data, centroid_data) / (np.linalg.norm(node_data) * 
                                                         np.linalg.norm(centroid_data))) if similarity_type == 'cosine' else np.linalg.norm(node_data - centroid_data)

    return new_node_attribute

def compute_nodes_attribute(cluster_file, centroid_type='mean', similarity_type='cosine'):
    # Load the cluster data from the text file
    cluster_data = np.loadtxt(cluster_file)

    # Extract the nodes
    # nodes = cluster_data[:-1]  # Exclude the last row which represents the centroid

    # Compute centroid from the cluster data by taking the mean or median
    centroid_data = np.mean(cluster_data, axis=0) if centroid_type == 'mean' else np.median(cluster_data, axis=0)

    # Compute the cosine similarity for each vector in cluster_data
    nodes_attribute = [1 - (np.dot(cluster_data[i], centroid_data) / (np.linalg.norm(cluster_data[i]) * np.linalg.norm(centroid_data)) )
                       for i in range(len(cluster_data))] if similarity_type == 'cosine' else [np.linalg.norm(cluster_data[i] - centroid_data) 
                                                                                               for i in range(len(cluster_data))]

    # Convert the list to a numpy array
    nodes_attribute = np.array(nodes_attribute)

    return nodes_attribute, centroid_data

def visualize_meas_central_tend(nodes_attributes, new_node_attributes, similarity_type='cosine'):
    nodes_attributes_1 = nodes_attributes[0]
    nodes_attributes_2 = nodes_attributes[1]
    nodes_attributes_3 = nodes_attributes[2]
    new_node_attribute_1 = new_node_attributes[0]
    new_node_attribute_2 = new_node_attributes[1]
    new_node_attribute_3 = new_node_attributes[2]
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Set the aspect ratio of the plot
    ax.view_init(elev=20, azim=-45)  # Set the viewing angle of the plot
    ax.dist = 12  # Set the distance between the plot and the camera

    # # Compute the mean and standard deviation of the nodes_attributes_1 array
    # mean = np.mean(nodes_attributes_1)
    # std = np.std(nodes_attributes_1)
    # # Generate the range of values for x axis in the 95% confidence interval
    # x = np.linspace(mean - 2*std, mean + 2*std, 100)
    # # Compute the probability density function using the mean and standard deviation
    # y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    # # Normalize the probability density function
    # y = y / (np.max(y)*2)
    # # Plot the normal distribution curve in the x and y axis of the 3D plot
    # ax.plot(x, y, np.zeros(len(x)), color='blue')#, label='Normal Distribution')

    # Compute the histogram of the nodes_attributes_1 array
    hist, bins = np.histogram(nodes_attributes_1, bins=50, density=True)
    bins = bins + (bins[1] - bins[0])/2  # Shift the bins to the center
    bins = bins[:-1] # Remove the last bin
    # Superior limit for the histogram
    limit_sup = 1 if similarity_type == 'cosine' else np.max(nodes_attributes_2)
    limit_sup = 0.5*limit_sup
    # Plot the histogram in the x and y axis of the 3D plot
    ax.plot(bins, hist*limit_sup / np.max(hist), np.zeros(len(bins)), c='royalblue', linewidth=0.5, alpha=0.7)
    # Draw the median of the histogram with a dotted line
    median_1 = np.median(nodes_attributes_1)
    # ax.plot([median_1, median_1], [0, 1], [0, 0], linestyle='--', color='cornflowerblue', linewidth=2.5, label='Median Cluster 1', alpha=0.7)
    # Draw the mode of the histogram with a dotted line
    mode_1 = bins[np.argmax(hist)]
    ax.plot([mode_1, mode_1], [0, limit_sup], [0, 0], linestyle='--', color='cornflowerblue', linewidth=2.5, label='Mode Cluster 1', alpha=0.7)
    # Normalize the histogram values
    hist = hist / np.sum(hist)
    # Find the index of the 95% percentile of the histogram
    index = np.where(np.cumsum(hist) >= 0.95)[0][0]
    # Draw the 95% percentile of the histogram with a dotted line
    percentile95_1 = bins[index]
    ax.plot([percentile95_1, percentile95_1], [0, limit_sup], [0, 0], linestyle='--', color='slategrey', linewidth=2.5, label='95% Percentile Cluster 1', alpha=0.7)

    # Plot the nodes from the first set
    ax.scatter(nodes_attributes_1, np.zeros(len(nodes_attributes_1)), np.zeros(len(nodes_attributes_1)), c='royalblue', label='Cluster 1')
    # Plot the new node in the axis of cluster 1
    ax.scatter(new_node_attribute_1, 0, 0, c='red', marker='x', s=100, label='New Node Similarity with Cluster 1')
    
    
    # # Compute the mean and standard deviation of the nodes_attributes_2 array
    # mean = np.mean(nodes_attributes_2)
    # std = np.std(nodes_attributes_2)
    # # Generate the range of values for x axis in the 95% confidence interval
    # y = np.linspace(mean - 2*std, mean + 2*std, 100)
    # # Compute the probability density function using the mean and standard deviation
    # z = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    # # Normalize the probability density function
    # z = z / (np.max(z)*2)
    # # Plot the normal distribution curve in the x and y axis of the 3D plot
    # ax.plot(np.zeros(len(y)), y, z, color='red')#, label='Normal Distribution')

    # Compute the histogram of the nodes_attributes_1 array
    hist, bins = np.histogram(nodes_attributes_2, bins=50, density=True)
    bins = bins + (bins[1] - bins[0])/2  # Shift the bins to the center
    bins = bins[:-1] # Remove the last bin
    # Superior limit for the histogram
    limit_sup = 1 if similarity_type == 'cosine' else np.max(nodes_attributes_3)
    limit_sup = 0.5*limit_sup 
    # Plot the histogram cure in the y and z axis of the 3D plot
    ax.plot(np.zeros(len(bins)), bins, hist*limit_sup / np.max(hist), c='firebrick', linewidth=0.5, alpha=0.7)
    # Draw the median of the histogram with a dotted line
    median_2 = np.median(nodes_attributes_2)
    # ax.plot([0, 0], [median_2, median_2], [0, 1], linestyle='--', color='indianred', linewidth=2.5, label='Median Cluster 2', alpha=0.7)
    # Draw the mode of the histogram with a dotted line
    mode_2 = bins[np.argmax(hist)]
    ax.plot([0, 0], [mode_2, mode_2], [0, limit_sup], linestyle='--', color='indianred', linewidth=2.5, label='Mode Cluster 2', alpha=0.7)
    # Normalize the histogram values
    hist = hist / np.sum(hist)
    # Find the index of the 95% percentile of the histogram
    index = np.where(np.cumsum(hist) >= 0.95)[0][0]
    # Draw the 95% percentile of the histogram with a dotted line
    percentile95_2 = bins[index]
    ax.plot([0, 0], [percentile95_2, percentile95_2], [0, limit_sup], linestyle='--', color='slategrey', linewidth=2.5, label='95% Percentile Cluster 2', alpha=0.7)

    # Plot the nodes from the second set
    ax.scatter(np.zeros(len(nodes_attributes_2)), nodes_attributes_2, np.zeros(len(nodes_attributes_2)), c='firebrick', label='Cluster 2')
    # Plot the new node in the axis of cluster 2
    ax.scatter(0, new_node_attribute_2, 0, c='red', marker='x', s=100, label='New Node Similarity with Cluster 2')

    # # Compute the mean and standard deviation of the nodes_attributes_2 array
    # mean = np.mean(nodes_attributes_3)
    # std = np.std(nodes_attributes_3)
    # # Generate the range of values for x axis in the 95% confidence interval
    # z = np.linspace(mean - 2*std, mean + 2*std, 100)
    # # Compute the probability density function using the mean and standard deviation
    # x = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    # # Normalize the probability density function
    # x = x / (np.max(x)*2)
    # # Plot the normal distribution curve in the x and y axis of the 3D plot
    # ax.plot(x, np.zeros(len(z)), z, color='green')#, label='Normal Distribution')

    # Compute the histogram of the nodes_attributes_1 array
    hist, bins = np.histogram(nodes_attributes_3, bins=50, density=True)
    bins = bins + (bins[1] - bins[0])/2  # Shift the bins to the center
    bins = bins[:-1] # Remove the last bin
    # Superior limit for the histogram
    limit_sup = 1 if similarity_type == 'cosine' else np.max(nodes_attributes_1)
    limit_sup = 0.5*limit_sup
    # Plot the histogram curve in the z and x axis of the 3D plot
    ax.plot(hist*0.5*limit_sup / np.max(hist), np.zeros(len(bins)), bins, c='darkolivegreen', linewidth=0.5, alpha=0.7)
    # Draw the median of the histogram with a dotted line
    median_3 = np.median(nodes_attributes_3)
    # ax.plot([0, 1], [0, 0], [median_3, median_3], linestyle='--', color='olivedrab', linewidth=2.5, label='Median Cluster 3', alpha=0.7)
    # Draw the mode of the histogram with a dotted line
    mode_3 = bins[np.argmax(hist)]
    ax.plot([0, limit_sup], [0, 0], [mode_3, mode_3], linestyle='--', color='olivedrab', linewidth=2.5, label='Mode Cluster 3', alpha=0.7)
    # Normalize the histogram values
    hist = hist / np.sum(hist)
    # Find the index of the 95% percentile of the histogram
    index = np.where(np.cumsum(hist) >= 0.95)[0][0]
    # Draw the 95% percentile of the histogram with a dotted line
    percentile95_3 = bins[index]
    ax.plot([0, limit_sup], [0, 0], [percentile95_3, percentile95_3], linestyle='--', color='slategrey', linewidth=2.5, label='95% Percentile Cluster 3', alpha=0.7)

    # Plot the nodes from the third set
    ax.scatter(np.zeros(len(nodes_attributes_3)), np.zeros(len(nodes_attributes_3)), nodes_attributes_3, c='darkolivegreen', label='Cluster 3')
    # Plot the new node in the axis of cluster 3
    ax.scatter(0, 0, new_node_attribute_3, c='red', marker='x', s=100, label='New Node Similarity with Cluster 3')

    # Set labels and legend
    ax.set_xlabel('Distance with cluster 1 centroid')
    ax.set_ylabel('Distance with cluster 2 centroid')
    ax.set_zlabel('Distance with cluster 3 centroid')
    # ax.legend()
    # Show the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    percentile95 = [percentile95_1, percentile95_2, percentile95_3]
    mode = [mode_1, mode_2, mode_3]
    median = [median_1, median_2, median_3]
    return percentile95, mode, median
    
def find_similar_class(new_feature_vector, file_paths, class_labels):
    similarity_type = 'cosine'#'euclidean'#'cosine'
    nodes_attributes = []
    new_node_attributes = []
    for file_path in file_paths:
        nodes_attribute, centroid_vector = compute_nodes_attribute(file_path, centroid_type='mean', similarity_type=similarity_type)
        new_node_attribute = compute_new_node_attribute(new_feature_vector, centroid_vector, similarity_type=similarity_type)
        nodes_attributes.append(nodes_attribute)
        new_node_attributes.append(new_node_attribute)
    percentile95, mode, median = visualize_meas_central_tend(nodes_attributes, new_node_attributes, similarity_type=similarity_type)
    # similarity_list = [new_node_attribute if new_node_attribute < percentile95[index] else 10000 for index, new_node_attribute in enumerate(new_node_attributes)]
    # class_index = similarity_list.index(min(similarity_list))
    class_index = new_node_attributes.index(min(new_node_attributes))
    return class_index, class_labels[class_index]

# Example usage
file_paths = [
    'L_vectors_last.txt',
    'C_vectors_last.txt',
    'None_vectors_last.txt'
]

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data_string = file.read()
        data_array = np.fromstring(data_string, sep=' ')
        print(data_array)

new_vector_data = "-0.201475 -0.103312 -0.360104 -0.191235 -0.364391 -0.275642 -0.358644 -0.374865 0.033989 1.209823 -0.198043 -0.222960 -0.086328 -0.007848 -0.340227 -0.315525 -0.000000 -0.371418 0.654324 0.204742 -0.128994 0.055450 1.759473 -0.224591 -0.000000 1.086074 -0.000000 -0.331882 -0.000000 -0.234655 -0.344115 0.109355 -0.016075 -0.322082 -0.000000 0.078550 -0.125295 -0.000000 -0.358003 -0.199792 -0.000000 0.728785 -0.327007 -0.249808 1.579939 -0.318685 -0.350033 2.874880 0.317017 0.414763 -0.278322 0.693575 -0.317227 -0.304403 0.161599 -0.212129 -0.374527 -0.184429 -0.159322 -0.060744 -0.374982 -0.359984 -0.287779 -0.126968 -0.162356 1.187629 -0.185324 -0.297075 1.333567 -0.225773 -0.362765 -0.072790 -0.359954 1.281003 -0.312344 -0.000000 2.428267 0.253656 -0.359627 -0.272219 0.672933 -0.302504 0.050677 -0.302241 -0.373705 -0.174129 -0.307457 -0.207840 -0.000000 -0.374357 -0.373300 -0.115338 0.379744 -0.000000 -0.098941 0.549153 2.872794 0.100999 -0.000000 -0.283395 -0.345322 -0.000000 -0.209229 -0.061577 -0.360040 -0.367096 -0.257764 0.177403 -0.373021 -0.296723 -0.181006 -0.317950 0.785728 0.245179 -0.363547 0.563700 1.102608 -0.079277 -0.355563 -0.000000 -0.000000 -0.351623 -0.304012 -0.106595 0.030380 0.716785 -0.359160 0.482171 -0.012781 -0.000000 -0.374740 -0.160714 -0.003920 -0.107919 0.018420 -0.310960 -0.246032 -0.207074 -0.041863 -0.000000 0.000070 -0.265610 -0.337938 1.425287 -0.090063 -0.193091 -0.215143 -0.216632 -0.341750 -0.354345 -0.262120 -0.366593 -0.098832 0.499767 -0.367281 -0.230710 -0.208214 -0.029728 0.249652 1.298118 -0.004880 -0.000000 -0.349043 0.199399 -0.068927 0.450321 -0.368427 -0.048890 0.233578 -0.209573 -0.000000 0.193459 -0.374997 1.294936 -0.355843 -0.300277 0.166110 0.356610 0.381531 -0.265843 0.448810 0.669555 -0.327994 -0.368423 -0.000000 -0.341234 -0.163530 -0.287275 0.142338 0.225771 1.717502 1.471497 0.307165 -0.135772 0.044500 -0.278457 -0.342211 -0.337142 -0.107814 -0.041339 -0.285761 -0.308779 -0.134560 1.402296 -0.184638 -0.251773 -0.116991 -0.344092 -0.010024 0.437946 -0.374491 -0.373198 0.106021 1.805426 -0.342659 -0.325531 0.354214 -0.319196 -0.272956 -0.000000 -0.335133 -0.327174 -0.304198 0.372305 0.029474 -0.268322 0.884644 -0.374993 0.317995 -0.263585 0.205900 0.844899 -0.368150 0.055007 1.328387 -0.283702 -0.220736 -0.355051 -0.371836 -0.224608 0.153205 -0.228418 -0.163864 -0.338295 -0.360312 -0.117162 -0.363912 -0.083506 0.295108 0.329207 0.352791 -0.246807 -0.000000 -0.292088 -0.358806 -0.318251 -0.368365 0.391572 -0.361664 -0.318887 -0.039556 1.253961 -0.355609 -0.023838 -0.330388 -0.358706 1.001240 0.969440 -0.331343 1.696250 -0.197305 -0.312965 -0.298478 0.249715 0.914635 -0.341069 -0.143143 -0.071376 -0.000000 -0.160363 -0.373189 -0.290410 -0.374962 -0.093668 -0.234544 -0.214482 -0.326678 1.205201 1.261866 0.422951 0.935334 -0.366058 -0.120080 -0.329367 -0.000000 -0.166927 -0.322865 0.881975 -0.000000 0.142422 0.057491 0.151830 -0.000000 -0.257210 1.267332 0.125364 -0.355837 -0.311224 -0.359101 -0.167823 1.910404 1.626656 -0.372044 0.351262 1.631008 1.117691 -0.000000 0.206753 -0.009489 0.176659 1.402992 -0.263167 0.357376 -0.255910 -0.373360 0.994439 -0.368947 0.082405 -0.279646 -0.374749 -0.260750 0.215362 -0.216982 0.009543 0.768581 -0.374650 -0.319155 -0.150631 -0.040338 -0.015405 -0.365435 -0.361595 -0.000000 -0.220620 -0.194789 -0.039715 -0.241835 -0.374315 -0.092083 0.248101 -0.184036 0.772070 -0.116520 3.656714 0.390946 0.450240 -0.319602 -0.359453 1.907818 -0.223841 0.216786 -0.000000 -0.000000 -0.000000 -0.211973 0.400054 -0.367939 -0.163514 -0.374796 -0.372396 0.053323 -0.374890 0.171994 -0.246513 0.405284 0.006725 0.145318 -0.000000 0.818984 -0.348973 -0.000000 -0.201771 2.017614 -0.000000 1.217269 -0.283332 0.139009 -0.304325 0.360144 0.344826 -0.365640 1.555398 -0.047315 -0.360194 -0.090919 -0.247143 1.018213 -0.296999 1.255360 -0.184855 1.900761 -0.358845 -0.331971 -0.108313 -0.213827 0.063497 -0.344853 -0.270248 -0.352093 0.010387 -0.000000 -0.136404 -0.227765 -0.301373 1.387419 0.252611 -0.178910 -0.047364 -0.000000 3.323664 0.508030 0.877383 1.508428 0.314966 -0.274792 -0.320078 -0.229522 -0.196225 -0.315199 -0.328344 -0.245448 -0.365241 0.073971 -0.366072 -0.370668 0.825096 -0.132277 -0.000000 1.108955 -0.085917 -0.000000 -0.158415 -0.109598 0.066963 1.080331 -0.301138 0.048247 -0.372382 0.325915 -0.239044 -0.359463 -0.373029 -0.368401 -0.178861 -0.362434 0.107584 -0.374979 -0.141912 -0.209954 -0.348777 0.097657 -0.043916 -0.361484 -0.286864 0.227167 0.238506 0.492512 0.076238 -0.372268 -0.256750 0.383992 -0.370029 -0.000000 -0.173578 -0.308987 0.089292 -0.000000 -0.336832 -0.051462 0.129847 -0.356787 -0.032045 -0.374814 -0.000000 -0.286921 0.075790 -0.058433 -0.275391 -0.354798 -0.059652 -0.372358 -0.173556 -0.373512 -0.122469 -0.225909 -0.128312 -0.294019 0.398065 0.037497 -0.169562 0.627267 -0.366566 0.184303 -0.093212 0.224421 -0.373833 0.004685 -0.364212 -0.226537 -0.066244 -0.148091 -0.285555 -0.348376 0.017886 -0.374998 0.199021 0.885534 0.044220 0.434699 0.647720 1.054661 0.674115 -0.374071 -0.369234 -0.231251 -0.120832 0.215080 -0.348272 0.815236 -0.374502 -0.227963 0.153897 -0.363248 -0.149888 -0.232344 -0.000663 -0.097071 -0.298683 -0.202802 -0.270079 0.349869 0.053838 0.113976 -0.241585 0.499985 -0.147117 -0.353680 -0.271389 -0.052652 -0.362756 -0.037520 0.901941 -0.352658 0.225080 -0.317430 0.364072 -0.277054 -0.317906 -0.367099 -0.345463 -0.333371 0.121853 0.174125 0.961159 -0.221686 -0.342154 -0.299983 -0.000000 -0.371496 1.384894 -0.126971 -0.370049 -0.083115 -0.286168 -0.301529 0.131878 -0.295824 -0.000000 1.822623 1.137470 -0.000000 -0.000000 -0.000000 -0.360386 0.228729 -0.345293 0.305484 -0.210033 -0.295790 -0.095957 -0.000000 0.017156 -0.328458 -0.263753 -0.374515 -0.303409 -0.245815 0.987803 -0.000000 -0.097917 -0.221362 -0.289656 -0.249879 -0.200499 0.463699 -0.293313 -0.276205 -0.164662 0.284428 1.073413 -0.373592 -0.358816 -0.336827 -0.255881 0.050440 -0.257789 -0.000000 -0.316042 -0.363208 1.468711 0.112194 1.575417 -0.284111 1.304811 -0.000000 0.403618 0.368140 0.465620 -0.362450 -0.109134 -0.296862 -0.369155 1.286473 -0.309426 -0.000000 0.506617 -0.343407 4.130911 0.253103 0.956467 0.834289 -0.359681 0.155482 0.112365 -0.254886 -0.093043 0.581079 0.170903 -0.259524 -0.320566 0.268751 0.111708 -0.336435 -0.236353 0.395789 -0.374674 0.431476 0.142059 0.252779 -0.366239 -0.326577 -0.000000 3.120574 0.636197 -0.372950 -0.371433 -0.013329 -0.000000 -0.072984 -0.075605 -0.104345 1.876117 -0.039118 -0.298724 -0.278233 -0.364177 -0.125755 -0.374497 -0.000000 -0.242677 -0.230822 -0.039179 0.822060 -0.315621 -0.115051 -0.309131 -0.235068 1.904608 -0.000000 -0.038380 0.375602 0.977443 1.057315 0.339337 -0.234919 0.286560 -0.214297 -0.229695 -0.292501 0.170793 -0.368274 -0.000000 -0.014999 -0.308272 -0.365107 0.264447 0.951206 -0.281903 -0.050080 0.610967 -0.336629 -0.374478 -0.237564 -0.009059 0.961959 0.508297 0.480992 0.559988 0.177761 -0.374302 0.257746 -0.230914 -0.336960 -0.005780 -0.348365 -0.334968 -0.000000 -0.365310 -0.000000 -0.374680 -0.171593 1.860418 -0.112559 1.036044 -0.263430 -0.141090 0.628726 1.078416 -0.324849 0.017303 0.138298 0.704607 0.063710 -0.000000 -0.350285 -0.349795 -0.361115 0.433085 -0.106565 -0.051849 -0.273575 -0.369967 2.000594 1.976738 -0.118661 -0.374248 0.167597 -0.342417 -0.288848 0.339850 1.100641 -0.348533 -0.373483 0.118450 -0.297956 -0.037996 1.572132 0.583339 -0.362974 -0.374741 0.131296 -0.000000 0.061942 -0.312616 1.084501 -0.365660 3.063718 0.048457 -0.000000 -0.372651 -0.342880 0.069374 1.658167 0.151136 -0.164070 0.356806 -0.137239 -0.044367 -0.333564 0.135520 -0.374828 0.143263 0.181541 -0.029273 -0.000000 1.014627 -0.338774 2.346517 -0.000000 0.003622 -0.258517 -0.153441 -0.214782 -0.062422 0.963904 -0.321864 -0.366918 -0.015210 0.908581 -0.374840 -0.364310 -0.000000 -0.037887 -0.371485 -0.314127 0.352154 1.172765 0.117214 1.020450 -0.373380 -0.324254 -0.000000 -0.369528 -0.002495 1.609640 -0.343003 -0.349821 -0.334136 -0.274090 -0.228505 -0.365332 0.074878 1.380430 1.128532 -0.367885 -0.187480 -0.286151 -0.373384 -0.185140 0.571669 -0.161825 0.287351 0.295285 -0.291428 0.212321 -0.207010 -0.190069 -0.311378 -0.372134 0.773158 -0.354114 0.466739 0.501110 -0.304973 -0.138608 0.048174 -0.374503 0.480266 -0.374650 -0.374908 -0.000000 -0.047128 -0.373524 -0.000000 -0.364348 0.181714 -0.222839 -0.083831 -0.373314 0.169308 -0.269984 0.367155 0.363611 -0.289380 1.248796 -0.343973 0.023745 -0.317721 0.857489 -0.205540 -0.297047 0.430507 -0.314238 -0.000000 -0.344014 -0.311310 -0.122252 -0.374115 0.154840 -0.347351 1.100981 0.483002 -0.222306 -0.052018 -0.298675 0.942118 0.595166 -0.320056 1.310600 -0.353427 -0.283626 0.379544 0.003554 -0.142608 -0.177129 -0.301127 0.485905 -0.304549 -0.351046 -0.180092 -0.000000 -0.000000 -0.374950 -0.067917 0.088749 1.493322 -0.136589 0.239986 -0.350218 0.244317 2.862250 0.023654 -0.000000 -0.372950 -0.256371 -0.288564 -0.055913 -0.176841 -0.000000 -0.318914 0.190521 -0.288466 -0.218554 -0.239624 0.646471 -0.248969 -0.341825 0.082633 -0.196123 1.021387 -0.297386 -0.000000 -0.294833 0.191719 -0.258162 -0.000000 -0.076220 0.228819 -0.319034 0.165643 1.927406 -0.000000 0.438733 -0.316355 -0.337471 -0.373532 -0.237710 -0.147181 0.600158 -0.360464 -0.081504 -0.286149 -0.189887 1.206110 0.361573 -0.372692 0.690615 -0.033988 1.648317 -0.320597 -0.000000 -0.172520 -0.181815 -0.137926 -0.000000 -0.120660 -0.372468 -0.219941 -0.324261 -0.295447 -0.205558 0.257279 0.517529 0.381776 -0.000000 0.258255 -0.374694 -0.200013 -0.364413 -0.359062 -0.337703 -0.374124 -0.233276 0.131720 -0.101407 -0.157411 -0.137376 -0.345813 -0.336112 0.986544 0.162843 -0.363242 0.750822 -0.258824 1.245682 -0.161847 -0.357435 -0.300487 -0.349506 -0.369404 -0.184289 -0.373351 -0.368496 0.027642 0.050192 -0.367406 -0.000000 -0.374762 -0.000000 -0.000000 1.406390 -0.369367 -0.000000 0.850218 "
new_vector_array = np.fromstring(new_vector_data, sep=' ')
class_labels = ['L', 'C', 'None']
# cluster_file = 'L_vectors_001.txt'
class_index, class_label = find_similar_class(new_vector_array, file_paths, class_labels)
print('The new feature vector belongs to class:', class_label)

# def visualize_cluster(cluster_file):
#     # Load the cluster data from the text file
#     cluster_data = np.loadtxt(cluster_file)

#     # Extract the nodes
#     # nodes = cluster_data[:-1]  # Exclude the last row which represents the centroid
#     # Compute centroid from the cluster data by taking the mean
#     centroid_data = np.mean(cluster_data, axis=0)  # Compute the centroid
#     # Compute the median vector of the cluster data
#     # centroid_data = np.median(cluster_data, axis=0)

#     # Compute the cosine similarity for each vector in cluster_data
#     nodes_attribute = [np.dot(cluster_data[i], centroid_data) / (np.linalg.norm(cluster_data[i]) * np.linalg.norm(centroid_data)) for i in range(len(cluster_data))]

#     # Convert the list to a numpy array
#     nodes_attribute = np.array(nodes_attribute)

#     # Create a random centroid position within a 5x5 area
#     centroid = (np.random.uniform(0, 5), np.random.uniform(0, 5))
#     centroid = (0.0, 0.0)

#     # Create a list of nodes positions
#     nodes = []
#     for i in range(len(nodes_attribute)):
#         # Compute the norm of the nodes_attribute value
#         norm = nodes_attribute[i]
#         # Generate a random angle between 0 and 2*pi
#         angle = np.random.uniform(0, 2*np.pi)
#         # Compute the x and y coordinates relative to the centroid
#         x = centroid[0] + norm * np.cos(angle)
#         y = centroid[1] + norm * np.sin(angle)
#         # Add the (x, y) position to the list of nodes
#         nodes.append((x, y))
    

#     # Convert the list of nodes to a numpy array
#     nodes = np.array(nodes)

#     # Plot the nodes
#     plt.scatter(nodes[:, 0], nodes[:, 1], s=50, c='blue', label='Nodes', marker='o')

#     # Plot the centroid
#     plt.scatter(centroid[0], centroid[1], s=100, c='red', label='Centroid', marker='o')

#     # Plot the unitary circle
#     theta = np.linspace(0, 2*np.pi, 100)
#     x_circle = np.cos(theta)
#     y_circle = np.sin(theta)
#     plt.plot(x_circle, y_circle, color='black', linestyle='--', label='Unitary Circle')

#     # Add labels and legend
#     plt.xlabel('Distance component in x', fontsize=14)
#     plt.ylabel('Distance component in y', fontsize=14)
#     # # Remove ticks labels
#     # plt.xticks([])
#     # plt.yticks([])
#     # Set the ticks to show by every integer unit
#     plt.xticks(np.arange(int(min(nodes[:, 0])) - 1, int(max(nodes[:, 0])) + 2, 1), fontsize=11)
#     plt.yticks(np.arange(int(min(nodes[:, 1])) - 1, int(max(nodes[:, 1])) + 2, 1), fontsize=11)
#     # # Set the ticks to show by every integer unit
#     # plt.xticks(np.arange(int(min(nodes[:, 0])) - 1, int(max(nodes[:, 0])) + 2, 1))
#     # plt.yticks(np.arange(int(min(nodes[:, 1])) - 1, int(max(nodes[:, 1])) + 2, 1))
#     plt.grid(True, color='gray', alpha=0.3)
#     # legend and title
#     legend = plt.legend(fontsize='12', loc='upper left')#loc ='upper right')
#     # Get the bounding box of the legend relative to the figure
#     bbox = legend.get_window_extent().transformed(plt.gca().transData.inverted())
#     # Calculate width and height of the legend box
#     x_inf_lim, x_sup_lim = plt.xlim()#ax.get_xlim()
#     y_inf_lim, y_sup_lim = plt.ylim()#ax.get_ylim()
#     legend_width = bbox.width/(x_sup_lim-x_inf_lim)
#     legend_height = bbox.height/(y_sup_lim-y_inf_lim)
#     print('legend_width =', legend_width)
#     print('legend_height =', legend_height)
#     # The y position for centering the legend
#     legend_y_pos = 0.5*(1+legend_height)
#     print ('legend_y_pos =',legend_y_pos)
#     legend.set_bbox_to_anchor((1.05, legend_y_pos))

#     # Adjust the figure size with rcParams
#     plt.rcParams['figure.figsize'] = [9, 6]
#     # Adjust the figure margins
#     plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

#     # Show the plot
#     plt.show()
