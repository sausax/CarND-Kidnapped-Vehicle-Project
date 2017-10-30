#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cstdlib>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double stds[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(0, stds[0]);
	
	// TODO: Create normal distributions for y and theta.
	normal_distribution<double> dist_y(0, stds[1]);
	
	normal_distribution<double> dist_theta(0, stds[2]);

	for(int i=0;i<num_particles;i++){
		Particle p;
		p.id = i;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	//cout << "Inside prediction step " << endl;
	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(0, std_pos[0]);
	
	// TODO: Create normal distributions for y and theta.
	normal_distribution<double> dist_y(0, std_pos[1]);
	
	normal_distribution<double> dist_theta(0, std_pos[2]);


	for(int i=0;i<num_particles;i++){
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		if (fabs(yaw_rate) < 0.0001) {  
      		particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
      		particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);	
    	}else{
			particles[i].x += (velocity/yaw_rate) * (sin(theta + yaw_rate*delta_t) - sin(theta)) + dist_x(gen);
			particles[i].y += (velocity/yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t)) + dist_y(gen);
			particles[i].theta += yaw_rate*delta_t + dist_theta(gen);	
    	} 
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//cout << "In dataAssociation" << endl;

	for(LandmarkObs &obs: observations){
		double min_dist = dist(predicted[0].x, predicted[0].y, obs.x, obs.y);
		obs.id = 0;

		for(int i=1;i<predicted.size();i++){
			auto curr = predicted[i];

			double curr_dist = dist(curr.x, curr.y, obs.x, obs.y);

			if(curr_dist < min_dist){
				min_dist = curr_dist;
				obs.id = i;
			}
		}
	}
	//cout << "dataAssociation complete" << endl;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//cout << "In updateWeights" << endl;

	for(int i=0;i<num_particles;i++){
		// Convert landmark coordinates from local to global map coordinates
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		vector<LandmarkObs> transformed;
		for(LandmarkObs o: observations){
			LandmarkObs new_observation;
			new_observation.x = o.x*cos(theta) - o.y*sin(theta) + x;
			new_observation.y = o.x*sin(theta) + o.y*cos(theta) + y;
			transformed.push_back(new_observation);
		}

		vector<LandmarkObs> predicted;
		for(auto map_landmark: map_landmarks.landmark_list){
			float d = dist(map_landmark.x_f, map_landmark.y_f, x, y);
			if(d <= sensor_range){
				predicted.push_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
			}
		}
		// Weight is the product of Gaussian probabilities of nearby observed landmarks
		dataAssociation(predicted, transformed);

		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

		particles[i].weight = 1;
		for(int j=0;j<transformed.size();j++){

 
			LandmarkObs closest_landmark = predicted[transformed[j].id];
			double mu_x = closest_landmark.x;
			double mu_y = closest_landmark.y;

			double x_obs = transformed[j].x;
			double y_obs = transformed[j].y;

			// calculate exponent
			double exponent= (pow((x_obs - mu_x),2))/(2 * pow(sig_x, 2)) + (pow((y_obs - mu_y),2))/(2 * pow(sig_y, 2));

			// calculate weight using normalization terms and exponent
			particles[i].weight *= gauss_norm * exp(-exponent);
		}
		// Update particle weight	
	}

	//cout << "updateWeights complete" << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//cout << "In resample" << endl;

	vector<Particle> new_particles;

	// Add all weights
	vector<double> wts;
	double sum_wt = 0;
	for(auto p: particles){
		sum_wt = p.weight;
		wts.push_back(p.weight);
	}


    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(wts.begin(), wts.end());
    for(int i=0;i<wts.size();i++){
    	int indx = d(gen);
    	new_particles.push_back(particles[indx]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
