#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace std;

random_device rd;
mt19937 gen(rd());

class Bandit {
private:
  vector<double> q_true;
  normal_distribution<double> reward_noise;
  normal_distribution<double> walk_noise;

public:
  Bandit() : reward_noise(0.0, 1.0), walk_noise(0.0, 0.01) {
    q_true.resize(10, 0.0);
  }

  void random_walk() {
    for (auto& q : q_true) {
      q+=walk_noise(gen);
    }
  }

  double get_reward(int action) {
    return q_true[action] + reward_noise(gen);
  }

  int optimal_action() const {
    return distance(q_true.begin(), max_element(q_true.begin(), q_true.end()));
  }

  void reset() {
    fill(q_true.begin(), q_true.end(), 0.0);
  }

};

class Agent {
protected:
  vector<double> Q;
  double epsilon;
  bernoulli_distribution bern;

public:
  Agent(double eps) : epsilon(eps), bern(1 - eps) {
    Q.resize(10, 0.0);
  }

  virtual ~Agent() = default;
  virtual void update(int action, double reward) = 0;

  int choose_action() {
    if (bern(gen)) {
      uniform_int_distribution<int> unif(0, 9);
      return unif(gen);
    }
    return distance(Q.begin(), max_element(Q.begin(), Q.end()));
  }

  virtual void reset() {
    fill(Q.begin(), Q.end(), 0.0);
  }

};

class SampleAverageAgent : public Agent {
  vector<int> counts;

public:
  SampleAverageAgent(double eps) : Agent(eps) {
    counts.resize(10, 0);
  }

  void update(int action, double reward) override {
    counts[action]++;
    Q[action] += (reward - Q[action]) / counts[action];
  }

  void reset() override {
    Agent::reset();
    fill(counts.begin(), counts.end(), 0);
  }
};

class ConstantStepSizeAgent : public Agent {
  double alpha;

public:
  ConstantStepSizeAgent(double eps, double a) : Agent(eps), alpha(a) {}

  void update(int action, double reward) override {
    Q[action] += alpha * (reward - Q[action]);
  }

};

void run_exp() {
  const int num_runs = 2000;
  const int num_steps = 10000;

  Bandit bandit;
  SampleAverageAgent sa_agent(0.1);
  ConstantStepSizeAgent css_agent(0.1, 0.1);

  vector<double> sa_rewards(num_steps, 0.0);
  vector<double> css_rewards(num_steps, 0.0);
  vector<double> sa_optimal(num_steps, 0.0);
  vector<double> css_optimal(num_steps, 0.0);

  for (int run = 0; run < num_runs; ++run) {
    bandit.reset();
    sa_agent.reset();
    css_agent.reset();

    for (int step = 0; step < num_steps; ++step) {
      bandit.random_walk();
      int optimal_action = bandit.optimal_action();

      int sa_action = sa_agent.choose_action();
      double sa_reward = bandit.get_reward(sa_action);
      sa_agent.update(sa_action, sa_reward);
      sa_rewards[step] += sa_reward;
      sa_optimal[step] += (sa_action == optimal_action);

      int css_action = css_agent.choose_action();
      double css_reward = bandit.get_reward(css_action);
      css_agent.update(css_action, css_reward);
      css_rewards[step] += css_reward;
      css_optimal[step] += (css_action == optimal_action);

    }
  }

  transform(sa_rewards.begin(), sa_rewards.end(), sa_rewards.begin(), [num_runs](double d) {return d/ num_runs;});
  transform(css_rewards.begin(), css_rewards.end(), css_rewards.begin(), [num_runs](double d) {return d/ num_runs;});
  transform(sa_optimal.begin(), sa_optimal.end(), sa_optimal.begin(), [num_runs](double d) {return d/ num_runs * 100;});
  transform(css_optimal.begin(), css_optimal.end(), css_optimal.begin(), [num_runs](double d) {return d/ num_runs * 100;});

  ofstream reward_file("rewards.csv");
  ofstream optimal_file("optimal.csv");

  reward_file << "Step, SampleAverage, ConstantStepSize\n";
  optimal_file << "Step, SampleAverage, ConstantStepSize\n";

  for (int i = 0; i < num_steps; ++i) {
    reward_file << i << ',' << sa_rewards[i] << ',' << css_rewards[i] << '\n';
    optimal_file << i << ',' << sa_optimal[i] << ',' << css_optimal[i] << '\n';
  }

}

int main() {
  run_exp();
  std::cout << "1";
  return 0;
}


				      
