#include <pcl/point_cloud.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include "refinement.hpp"

void help() {
	std::cout << "parameter of the program: " << std::endl;
	std::cout << "argv[1]: ------the pcd/obj/ply file input\n";
	std::cout << "argv[2]: ------the points number wanna recover\n";
	std::cout << "argv[3]: ------the output filename\n";
}

int readFile(pcl::PointCloud<pcl::PointXYZ>::Ptr& input, const std::string& filename) {
	size_t pos = filename.find_last_of('.');
	if (pos == std::string::npos) {
		std::cerr << "bad file input\n";
		return -1;
	}
	else
	{
		std::string tmp = std::string(filename.begin() + pos + 1, filename.end());
		int state = -1;
		//std::cout << "file_format: " << tmp << std::endl;
		if (tmp == "obj")
			state = pcl::io::loadOBJFile(filename, *input);
		else if (tmp == "ply")
			state = pcl::io::loadPLYFile(filename, *input);
		else
			state = pcl::io::loadPCDFile(filename, *input);

		if (state == -1) {
			std::cerr << "bad file input\n";
			return -1;
		}
	}
}

int main(int argc, char** argv) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);

	if (argc < 5) {
		std::cerr << "argments not enough\n";
		help();
		return -1;
	}

	pcl::StopWatch time;
	std::string inputname(argv[1]);
	std::string outputname(argv[3]);
	const float alpha = 1.5f;
	const int max_nn = 15;
	size_t recover = static_cast<size_t>(atoi(argv[2]));

	if(readFile(input, inputname) == -1)
		std::cerr << "bad input\n";	

	std::cout << "before interpolate " << input->size() << std::endl;
	refinement refine(input, alpha, max_nn, recover);
	std::cout << "running time: " << time.getTimeSeconds() << std::endl;
	std::cout << "after interpolate " << input->size() << std::endl;

	if(pcl::io::savePCDFileASCII(outputname, *input) == -1)
		std::cerr << "save failed!\n";

	return 0;
}
