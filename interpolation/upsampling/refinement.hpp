#pragma once
#ifndef REFINEMENT_H
#define REFINEMENT_H
#include <pcl/io/auto_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/impl/accumulators.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/common/distances.h>
#include <pcl/features/normal_3d.h>
#include <queue>
#include <cmath>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <random>
#include <algorithm>


//#define DEBUG

struct hasher
{
public:
	size_t operator()(const pcl::PointXYZ& p) const {
		typedef size_t uint32;
		pcl::PointXYZ pos(int(p.x), int(p.y), int(p.z));
		uint32 hash = (((uint32)pos.x) * 73856093) ^ (((uint32)pos.y) * 19349663) ^ (((uint32)pos.z) * 83492791);
		return hash;
	}
};



struct query_node {

public:
	int index;
	float dist;

public:
	query_node(int index, float dist) :index(index), dist(dist) {

	}

	friend bool operator<(const query_node& a, const query_node& b) {
		return a.dist > b.dist;
	}
};

class refinement {

struct cmp {
public:
	bool operator()(const pcl::PointXYZ& a, const pcl::PointXYZ& b)const {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};


public:
	refinement() {}


	refinement(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const float alpha, int max_nn, size_t recover, bool thrid = false)
	{
		this->alpha = alpha;
		this->ptc = cloud;
		this->max_nn = max_nn;
		this->recoverpoints = recover;
		this->thrid = thrid;
		std::vector<int> indexs;
		std::vector<float> dist;
		search_idx.resize(ptc->size());
		for (size_t i = 0; i < ptc->size(); ++i) {
			search_idx[i] = i;
		}

		new_add = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
		s_index = boost::make_shared<std::vector<int>>();

		pcl::KdTreeFLANN<pcl::PointXYZ> kdsearch;
		kdsearch.setSortedResults(true);
		int step = 1;

		while(ptc->size() < recoverpoints) {
			kdsearch.setInputCloud(ptc);
			size_t last = ptc->size();
			this->resolution = computeCloudResolution(ptc);
			this->default_resolution = computeCloudResolution(ptc, max_nn);
			//std::cout << "resolution is: " << resolution << " with neighborhood: " << default_resolution << std::endl;

			if(step == 1)
				for (size_t j = 0; j < last ; ++j) {
#ifdef DEBUG
					if (j % 2000 == 0) {
						std::cout << "current points " << j << std::endl;
					}
#endif // DEBUG
					if(findNp(kdsearch, ptc->at(j), indexs, dist, step))
						projection(ptc->at(j), j, indexs);
				}

			else {
				int cnt = 0;
				int k = static_cast<int>(ptc->size() * 0.2);
				//std::cout << "select points: " << k << std::endl;
				fill_queue(kdsearch, indexs, dist, k);
				//std::cout << "select finish " << std::endl;
				auto top = top_k.top();
				while (!top_k.empty()) {
#ifdef DEBUG
					if (cnt % 2000 == 0) {
						std::cout << "current points " << cnt << std::endl;
					}
#endif // DEBUG
					cnt++;
					top = top_k.top();
					if (findNp(kdsearch, ptc->at(top.index), indexs, dist, step))
						projection(ptc->at(top.index), top.index, indexs);
					top_k.pop();
				}
			}

			step++;
			if (ptc->size() > recoverpoints)
				break;
			else
				interpolate();
			std::cout << "current pointcloud size: " << ptc->size() << std::endl;
			//std::cout << "two: " << two << " three: " << three << " four: " << four << " five: " << five << " six: " << six << std::endl;
			//break;
		}

	}


	virtual ~refinement() {}


	//get the center of gemotric
	inline pcl::PointXYZ getcog(const std::vector<pcl::PointXYZ>& points, bool thrid = false) {
		pcl::CentroidPoint<pcl::PointXYZ> center;
		pcl::PointXYZ ans;
		for (auto p : points) {
			center.add(p);
		}
		if (thrid) {
			Eigen::Vector3f a(points.front().data), b(points.back().data), c;
			c = ((b - a) / 3.f) + a;
			return pcl::PointXYZ(c.x(), c.y(), c.z());
		}
		else
			center.get(ans);
		return ans;
	}

	inline pcl::PointXYZ getcog(const std::vector<pcl::PointXYZ>&& points, bool thrid = false) {
		pcl::CentroidPoint<pcl::PointXYZ> center;
		pcl::PointXYZ ans;
		pcl::PointWithScale p;
		for (auto p : points) {
			center.add(p);
		}
		if (thrid) {
			Eigen::Vector3f a(points.front().data), b(points.back().data), c;
			c = ((b - a) / 3.f) + a;
			return pcl::PointXYZ(c.x(), c.y(), c.z());
		}
		else
			center.get(ans);
		return ans;
	}


	inline double getAngle(const pcl::PointXYZ& a, const pcl::PointXYZ& center,
		const pcl::PointXYZ& b) {

		Eigen::Vector3f _a(a.data);
		Eigen::Vector3f _b(center.data);
		Eigen::Vector3f _c(b.data);

		Eigen::Vector4f _d(_a.x(), _a.y(), _a.z(), 1.f), _e(_b.x(), _b.y(), _b.z(), 1.f), _f(_c.x(), _c.y(), _c.z(), 1.f);
		Eigen::Vector4f center_xy = cen * _e;
		Eigen::Vector4f begin_xy = cen * _d;
		Eigen::Vector4f last_xy = cen * _f;

		const double rate = (begin_xy[1] - center_xy[1]) / (begin_xy[0] - center_xy[0]);
		const double intercept = center_xy[1] - (rate * center_xy[0]);
		const double val = last_xy[1] - (rate * last_xy[0]) - intercept;
		//std::cout << "line equation is: y = " << rate << "x + " << intercept << std::endl;
		const double esp = 10e-4;

		if (val > 0) {
			Eigen::Vector3f a(begin_xy.x(), begin_xy.y(), 0.f);
			Eigen::Vector3f b(last_xy.x(), last_xy.y(), 0.f);
			return pcl::getAngle3D(a, b, true);
		}
		else if (val < 0) {
			Eigen::Vector3f a(begin_xy.x(), begin_xy.y(), 0.f);
			Eigen::Vector3f b(last_xy.x(), last_xy.y(), 0.f);
			return 360. - pcl::getAngle3D(a, b, true);
		}
		else {
			Eigen::Vector3f a(begin_xy.x(), begin_xy.y(), 0.f);
			Eigen::Vector3f b(last_xy.x(), last_xy.y(), 0.f);
			return pcl::getAngle3D(a, b, true);
		}
	}

	inline double get3DAngle(const pcl::PointXYZ& a,const pcl::PointXYZ& center,
		const pcl::PointXYZ& b) {
		Eigen::Vector3f _a(a.data) ;
		Eigen::Vector3f _b(center.data);
		Eigen::Vector3f _c(b.data);

		Eigen::Vector3f ab = _a - _b;
		Eigen::Vector3f cb = _c - _b;
		return pcl::getAngle3D(ab, cb, true);
	}

	//projection and sort the neighborhood
	void projection(const pcl::PointXYZ& center, const size_t index, std::vector<int>& idx)
	{
		if (idx.size() <= 1)
			return;

		pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>(*ptc, idx));
		pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>());

		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(ptc);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(1);
		Eigen::Vector4f ans;
		float curvature;
		ne.computePointNormal(*ptc, idx, ans, curvature);

		
		pcl::ModelCoefficients::Ptr mc(new pcl::ModelCoefficients());
		mc->values.resize(4);
		mc->values[0] = ans[0];
		mc->values[1] = ans[1];
		mc->values[2] = ans[2];
		mc->values[3] = ans[3];

		pcl::ProjectInliers<pcl::PointXYZ> project;
		project.setModelType(pcl::SACMODEL_PLANE);
		project.setInputCloud(temp);
		project.setModelCoefficients(mc);
		project.filter(*output);

		std::vector<std::pair<pcl::PointXYZ,int>> pts_idx;

		for (size_t i = 0; i < idx.size(); ++i) {
			pts_idx.push_back(std::make_pair(output->points.at(i), idx[i]));
		}

		sortNeighborhood(center, pts_idx);
		if (pts_idx.front().second != static_cast<int>(index))
			std::swap(pts_idx.front(), pts_idx.back());
		removeFarthestPoint(center, pts_idx);

		for (size_t i = 0; i < pts_idx.size(); ++i) {
			idx[i] = pts_idx[i].second;
		};
		CreatPolygonFan(center, pts_idx);

	}

	void removeFarthestPoint(const pcl::PointXYZ& p,std::vector<std::pair<pcl::PointXYZ, int>>& pi) {
		if (pi.size() <= 2)
			return;
		const double minangle = 180. / 8;
		pcl::PointXYZ begin = pi[1].first;
		for (size_t i = 2; i < pi.size(); ) {
			if (get3DAngle(begin, p, pi[i].first) < minangle) {
				float dista = pcl::euclideanDistance(p, begin);
				float distb = pcl::euclideanDistance(p, pi[i].first);
				
				if (dista > distb) { //begin is farthest 
					neighborhood.erase(neighborhood.begin() + (i -1)); //delete the point which angle less than 30
					pi.erase(pi.begin() + (i - 1));	//delete the idx too
				}
				else { //else case
					neighborhood.erase(neighborhood.begin() + i); //
					pi.erase(pi.begin() + i); //
				}
			}
			else
			{
				begin = pi[i].first;
				++i;
			}
		}
	}


	double
	computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, size_t knn = 2)
	{
		double res = 0.0;
		int n_points = 0;
		int nres;
		std::vector<int> indices(knn);
		std::vector<float> sqr_distances(knn);
		pcl::search::KdTree<pcl::PointXYZ> tree;
		tree.setInputCloud(cloud);

		for (size_t i = 0; i < cloud->size(); ++i)
		{
			if (!pcl_isfinite((*cloud)[i].x))
			{
				continue;
			}
			nres = tree.nearestKSearch(i, knn, indices, sqr_distances);//return :number of neighbors found 
			for(int i = 1; i < nres; ++i)
			{
				res += sqrt(sqr_distances[i]);
				++n_points;
			}
		}
		if (n_points != 0)
		{
			res /= n_points;
		}
		return res;
	}


	void sortNeighborhood(const pcl::PointXYZ& center, std::vector<std::pair<pcl::PointXYZ, int>>& pi) {
		const pcl::PointXYZ begin = pi.front().first;
		Eigen::Matrix4f temp;
		temp << 1.f, 0.f, 0.f, center.x,
			0.f, 1.f, 0.f, center.y,
			0.f, 0.f, 1.f, center.z,
			0.f, 0.f, 0.f, 1.f;
		auto inve = temp.inverse();
		cen = inve;
		sort(pi.begin(), pi.end(), [&](const std::pair<pcl::PointXYZ, int>& a, 
			const std::pair<pcl::PointXYZ, int>& b) {
			return getAngle(begin, center, a.first) < getAngle(begin, center, b.first);
		});
	}

	bool findNp(pcl::KdTreeFLANN<pcl::PointXYZ>& kdsearch, const pcl::PointXYZ& p, std::vector<int>& idx, std::vector<float>& dist, int step) {
		idx.clear();
		dist.clear();
		neighborhood.clear();
		
		float average_dist = 0.f;
		int points = 0;
		
		if (step == 1) {
			points = kdsearch.nearestKSearch(p, max_nn, idx, dist);
			if (std::floor(std::sqrt(dist[1])) == 0.) {
				idx.erase(idx.begin() + 1);
				dist.erase(dist.begin() + 1);
			}
			for (auto it = dist.begin(); it != dist.end(); ++it) {
				average_dist += std::sqrt(*it);
			}
			average_dist /= (points - 1);
			//if (average_dist < (default_resolution * 0.9)) {
			//	average_dist = std::sqrtf(dist.back()) * 0.85;
			//	idx.clear();
			//	dist.clear();
			//	points = kdsearch.radiusSearch(p, average_dist, idx, dist, max_nn);
			//}

			for (auto iter = idx.begin(); iter != idx.end(); ++iter) {
					neighborhood.push_back(ptc->points[*iter]);
				}
			return true;
		}
		else if(step > 1){
			points = kdsearch.nearestKSearch(p, max_nn >> 1, idx, dist);
			if (std::floor(std::sqrt(dist[1])) == 0.) {
				idx.erase(idx.begin() + 1);
				dist.erase(dist.begin() + 1);
			}
			for (auto iter = idx.begin(); iter != idx.end(); ++iter) {
				neighborhood.push_back(ptc->points[*iter]);
			}
			return true;
		}
	}

	void fill_queue(pcl::KdTreeFLANN<pcl::PointXYZ>& kdsearch, std::vector<int>& index, std::vector<float>& dist, int k=5000) {
		for (auto i = 0; i < ptc->size(); ++i) {
			int points = kdsearch.nearestKSearch(ptc->at(i), max_nn >> 1, index, dist);
			float average_dist = 0.f;
			std::for_each(dist.begin(), dist.end(), [&](auto& a) { average_dist += std::sqrt(a);});
			average_dist /= (points - 1);
			top_k.push({ i, average_dist });
			if (top_k.size() > k)
				top_k.pop();
		}
	}

	void processPolygonFan(const pcl::PointXYZ& center, const std::vector<pcl::PointXYZ>& localfan) {
		float resloute = resolution * 1.2;
		static size_t cnt = 0;
		if (localfan.size() == 2) {
			if ((pcl::euclideanDistance(localfan.front(), localfan.back())) > resloute) {
				two++;
				//pointset.push_back(getcog(localfan));
				pointset.insert(getcog(localfan, thrid));
				cnt = pointset.size();
			}
		}
		else if (localfan.size() == 3) {
			three++;
			std::vector<pcl::PointXYZ> pts = {getcog({ center,localfan[1]}, thrid), getcog({center, localfan.back()}, thrid)};
			for (auto pt : pts) {
				//pointset.push_back(pt);
				pointset.insert(pt);
				cnt = pointset.size();
			}
		}
		else if (localfan.size() > 3) {
			if (localfan.size() == 4)
				four++;
			else if (localfan.size() == 5)
				five++;
			else
				six++;

			std::vector<pcl::PointXYZ> pts = {getcog(localfan), getcog({center,localfan[1]}, thrid), getcog({ center, localfan.back()}, thrid) };
			for (auto pt : pts) {
				//pointset.push_back(pt);
				pointset.insert(pt);
				cnt = pointset.size();
			}
		}
	}


	void CreatPolygonFan(const pcl::PointXYZ& center, const std::vector<std::pair<pcl::PointXYZ, int>>& pi) {
		std::vector<pcl::PointXYZ> localfan;

		//dist beteewn p and center
		float begin = pcl::euclideanDistance(center, pi[1].first);
		localfan.push_back(center);
		localfan.push_back(ptc->points[pi[1].second]);

		std::vector<int> nums;
		nums.push_back(pi.front().second);
		nums.push_back(pi[1].second);
		
		for (size_t i = 2; i < pi.size(); ++i) {
			std::pair<pcl::PointXYZ, pcl::PointXYZ> cur = std::make_pair(center, pi[i].first);
			float dist = pcl::euclideanDistance(center, cur.second);
			if (dist > begin && localfan.size() <= 2) {
				localfan.push_back(ptc->points[pi[i].second]);//
				begin = dist;
				nums.push_back(pi[i].second);
			}
			else {
				localfan.push_back(ptc->points[pi[i].second]);
				nums.push_back(pi[i].second);
				auto pos = std::find_if(localfan.begin(), localfan.end(), [&](const pcl::PointXYZ& p) {
					return (center.x == p.x) && (center.y == p.y) && (center.z == p.z);
				});
				if (pos == localfan.end()) {
					nums.push_back(pi.front().second);
					localfan.push_back(center);
				}
				processPolygonFan(center, localfan);
				localfan.erase(localfan.begin()+1, localfan.end());
				localfan.push_back(ptc->points[pi[i].second]);
				nums.erase(nums.begin() + 1, nums.end());
				nums.push_back(pi[i].second);
				begin = dist;
			}
		}

		//localfan.push_back(pi[1].first);
		processPolygonFan(center, localfan);
	}

	void savefile(std::string&& savename) {
		 if (savename.find("pcd") != std::string::npos) {
			pcl::io::savePCDFile(savename, *ptc);
		}
		else if (savename.find("ply") != std::string::npos) {
			pcl::io::savePLYFile(savename, *ptc);
		}
		else
			return;
	}

	void interpolate() {
		//std::cout << "new pointset size: " << pointset.size() << std::endl;
		std::vector<int> index(5);
		std::vector<float> dist(5);
		
		new_add->clear();
		for (auto iter = pointset.begin(); iter != pointset.end(); ++iter) {
			new_add->push_back(*iter);
		}
		new_add->width = new_add->points.size();
		new_add->height = 1;

		addsearch.setInputCloud(new_add);
		addsearch.setSortedResults(true);
		double add_Resolute = computeCloudResolution(new_add);
		std::unordered_set<pcl::PointXYZ, hasher, cmp> fliter;
		
		for (auto it = pointset.begin(); it != pointset.end(); ++it) {
			if (ptc->size() < recoverpoints) {
				addsearch.nearestKSearch(*it, 3, index, dist);
				std::for_each(dist.begin(), dist.end(), [&](auto & a) { a = std::sqrt(a);});
				if (std::floor(dist[1]) == 0.) {
					if (std::floor(dist[2]) == 0.) {
						fliter.insert(getcog({ *it, new_add->points[index[1]], new_add->points[index[2]] }));
						pointset.erase(new_add->points[index[1]]);
						pointset.erase(new_add->points[index[2]]);
					}
					else if (dist[2] >= add_Resolute * 1.5) {
						fliter.insert(*it);
						//fliter.insert(new_add->points[index[2]]);
						pointset.erase(new_add->points[index[1]]);
						//pointset.erase(new_add->points[index[2]]);
					}
					else {
						fliter.insert(getcog({ *it, new_add->points[index[1]], new_add->points[index[2]] }));
						pointset.erase(new_add->points[index[1]]);
						pointset.erase(new_add->points[index[2]]);
					}
				}
				else
					fliter.insert(*it);
			}
			else
				break;
		}
		
		fliter.rehash(500009);
		for (auto it = fliter.begin(); it != fliter.end(); ++it)
			if (ptc->size() >= recoverpoints)
				break;
			else
				if(helper.count(*it) == 0) {
					ptc->push_back(*it);
					helper.insert(*it);
				}
		pointset.clear();
	}

	void addPoint() {
		for (auto it = pointset.begin(); it != pointset.end(); ++it) {
			if (ptc->size() < recoverpoints) {
				ptc->push_back(*it);
			}
			else
				break;
		}
	}

private:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	pcl::PointCloud<pcl::PointXYZ>::Ptr ptc;
	pcl::PointCloud<pcl::Normal> ptcnormal;
	std::vector<pcl::PointXYZ> neighborhood;
	std::vector<int> search_idx;
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_add;
	pcl::KdTreeFLANN<pcl::PointXYZ> addsearch;
	pcl::IndicesPtr s_index;
	std::priority_queue<query_node> top_k;
	//std::vector<pcl::PointXYZ> pointset;
	std::unordered_set<pcl::PointXYZ, hasher, cmp> helper;
	std::unordered_set<pcl::PointXYZ, hasher, cmp> pointset;
	Eigen::Matrix4f cen; //storeage the 4*4 transform matrix
	float alpha, radius;
	float resolution, default_resolution;
	int max_nn;
	int two, three, four, five, six;
	size_t recoverpoints;
	bool thrid;
};
#endif // !REFINEMENT_H

