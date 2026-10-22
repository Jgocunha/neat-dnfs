#include "solutions/serial_order_profile.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

#include "neat_tools/resource_paths.h"
#include "neat_tools/utils.h"
#include "solutions/serial_order_schedule.h"

namespace neat_dnfs
{
	namespace
	{
		const std::string inputField = "nf 1";
		const std::string outputField = "nf 2";
		// After a pulse ends, or the output falls silent after a recall, the reference's fields settle within about 50:
		// five time constants of the input, a detection field with tau 10.
		constexpr double settleTime = 50.0;
		// Where a field is flat at a moment, as the input between pulses, its level alone is compared: one unit off the
		// reference's halves the agreement.
		constexpr double flatTolerance = 1.0;

		// Three decimals are finer than the noise, which moves a sample by hundredths, and keep a saved field's profile
		// under a megabyte.
		double rounded(const double value)
		{
			return std::round(value * 1000.0) / 1000.0;
		}

		// How many events of the sequence are shown, from the first: SolutionConstants::profileEventCount, or all.
		size_t shownEvents()
		{
			return static_cast<size_t>(SolutionConstants::profileEventCount.value_or(
				static_cast<int>(serial_order_schedule::eventCount)));
		}

		// The whole sequence's reference is serial-order-profile.json; a shorter one's names its event count.
		std::filesystem::path referenceProfilePath(const size_t events)
		{
			const std::string name = events == serial_order_schedule::eventCount ? "serial-order-profile.json"
				: std::format("serial-order-profile-{}-event{}.json", events, events == 1 ? "" : "s");
			return paths::resourceRoot() / "config" / "references" / name;
		}

		size_t firstSampleFrom(const ActivationProfile& profile, const double time)
		{
			return static_cast<size_t>(std::ranges::lower_bound(profile.times, time - 1e-9) - profile.times.begin());
		}

		// When the reference's output first rises above threshold at @p position, from @p from on. Positions count from
		// 1, as Solution::clampedIndexForPosition takes them.
		double firstPeakAt(const ActivationProfile& reference, const double position, const double from)
		{
			const auto& output = reference.fields.at(outputField);
			const auto index = static_cast<size_t>(std::lround(position / DimensionConstants::dx) - 1);
			for (size_t sample = firstSampleFrom(reference, from); sample < output.size(); ++sample)
			{
				if (output[sample][index] > 0.0)
				{
					return reference.times[sample];
				}
			}
			throw std::runtime_error(std::format("The reference profile never peaks at {} after {}.", position, from));
		}

		// When the reference's output is next below threshold everywhere, from @p from on.
		double firstQuietFrom(const ActivationProfile& reference, const double from)
		{
			const auto& output = reference.fields.at(outputField);
			for (size_t sample = firstSampleFrom(reference, from); sample < output.size(); ++sample)
			{
				if (std::ranges::max(output[sample]) <= 0.0)
				{
					return reference.times[sample];
				}
			}
			throw std::runtime_error(std::format("The reference profile's output never falls silent after {}.", from));
		}

		// (1 + concordance) / 2: 1 for the reference's profile, 0.5 for an unrelated one, 0 for its mirror image.
		double agreementScore(const std::vector<double>& values, const std::vector<double>& reference,
			const double tolerance)
		{
			return (1.0 + tools::utils::concordance(values, reference, tolerance)) / 2.0;
		}

		// A field's output: its activation function at every position, near 0 below threshold and near 1 at a peak.
		std::vector<double> fieldOutput(const std::vector<double>& activation)
		{
			std::vector<double> output(activation.size());
			NeuralFieldConstants::activationFunction.apply(activation, output);
			return output;
		}

		// How closely the fields match the reference's at one sample: each field's agreement mapped to [0, 1], averaged.
		// With SolutionConstants::profileAlsoComparesFieldOutputs, a field scores the mean of its activation's agreement
		// and its output's, where a flat output is judged against one position's worth of output.
		double momentScore(const ActivationProfile& profile, const ActivationProfile& reference, const size_t sample)
		{
			const bool comparesOutputs = SolutionConstants::profileAlsoComparesFieldOutputs.value_or(false);
			double score = 0.0;
			for (const auto& [fieldName, samples] : profile.fields)
			{
				const auto& activation = samples[sample];
				const auto& referenceActivation = reference.fields.at(fieldName)[sample];
				double fieldScore = agreementScore(activation, referenceActivation, flatTolerance);
				if (comparesOutputs)
				{
					const double outputTolerance = 1.0 / static_cast<double>(activation.size());
					fieldScore = (fieldScore + agreementScore(fieldOutput(activation), fieldOutput(referenceActivation),
						outputTolerance)) / 2.0;
				}
				score += fieldScore;
			}
			return score / static_cast<double>(profile.fields.size());
		}

		// How active the output is at one sample: its own activation function at its highest point, near 0 below
		// threshold and near 1 for a peak.
		double outputActivity(const ActivationProfile& profile, const size_t sample)
		{
			std::vector<double> activity(1);
			NeuralFieldConstants::activationFunction.apply({ std::ranges::max(profile.fields.at(outputField)[sample]) },
				activity);
			return activity.front();
		}
	}

	nlohmann::json ActivationProfile::toJson() const
	{
		nlohmann::json fieldsJson = nlohmann::json::object();
		for (const auto& [fieldName, samples] : fields)
		{
			nlohmann::json rows = nlohmann::json::array();
			for (const auto& sample : samples)
			{
				nlohmann::json row = nlohmann::json::array();
				for (const double activation : sample)
				{
					row.push_back(rounded(activation));
				}
				rows.push_back(std::move(row));
			}
			fieldsJson[fieldName] = std::move(rows);
		}
		return nlohmann::json{ { "times", times }, { "fields", std::move(fieldsJson) } };
	}

	ActivationProfile ActivationProfile::fromJson(const nlohmann::json& json)
	{
		ActivationProfile profile;
		profile.times = json.at("times").get<std::vector<double>>();
		for (const auto& [fieldName, rows] : json.at("fields").items())
		{
			profile.fields[fieldName] = rows.get<std::vector<std::vector<double>>>();
		}
		return profile;
	}

	SerialOrderProfile::SerialOrderProfile(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Serial Order Profile";
	}

	SerialOrderProfile::SerialOrderProfile(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Serial Order Profile";
	}

	SolutionPtr SerialOrderProfile::clone() const
	{
		SerialOrderProfile solution(initialTopology);
		return std::make_shared<SerialOrderProfile>(solution);
	}

	SolutionPtr SerialOrderProfile::copy() const
	{
		SerialOrderProfile solution(initialTopology, phenotype);
		return std::make_shared<SerialOrderProfile>(solution);
	}

	ActivationProfile SerialOrderProfile::recordProfile()
	{
		buildPhenotype();
		try
		{
			ActivationProfile profile = runTrial();
			clearPhenotype();
			return profile;
		}
		catch (...)
		{
			clearPhenotype();
			throw;
		}
	}

	const ActivationProfile& SerialOrderProfile::referenceProfile()
	{
		// One reference per sequence length, read the first time it is needed; evaluations run on several threads.
		static std::mutex cacheMutex;
		static std::map<size_t, ActivationProfile> references;
		const size_t events = shownEvents();
		const std::lock_guard<std::mutex> lock(cacheMutex);
		if (const auto found = references.find(events); found != references.end())
		{
			return found->second;
		}
		std::ifstream file(referenceProfilePath(events));
		if (!file)
		{
			throw std::runtime_error(std::format("Cannot read the reference profile {}.",
				referenceProfilePath(events).generic_string()));
		}
		return references.emplace(events, ActivationProfile::fromJson(nlohmann::json::parse(file))).first->second;
	}

	const std::vector<SerialOrderProfile::Stage>& SerialOrderProfile::stages()
	{
		static std::mutex cacheMutex;
		static std::map<size_t, std::vector<Stage>> stagesByLength;
		const size_t events = shownEvents();
		const ActivationProfile& reference = referenceProfile();
		const std::lock_guard<std::mutex> lock(cacheMutex);
		if (const auto found = stagesByLength.find(events); found != stagesByLength.end())
		{
			return found->second;
		}

		using namespace serial_order_schedule;
		const auto shown = std::span(sequence).first(events);
		constexpr double response = pulseDuration + settleTime;
		std::vector<Stage> result{ { "trigger", cueTimes.front(), cueTimes.front() + response } };
		for (const auto& event : shown)
		{
			result.push_back({ std::format("event {}", event.label), event.onset, event.onset + response });
		}
		for (size_t cycle = 1; cycle < cueTimes.size(); ++cycle)
		{
			std::vector<std::pair<std::string, double>> starts{ { "X", cueTimes[cycle] } };
			for (const auto& event : shown)
			{
				starts.emplace_back(std::format("recall {}", event.label),
					firstPeakAt(reference, event.position, starts.back().second));
			}
			const double silence = firstQuietFrom(reference, starts.back().second);
			starts.emplace_back("release", silence);
			for (size_t stage = 0; stage < starts.size(); ++stage)
			{
				const double end = stage + 1 < starts.size() ? starts[stage + 1].second : silence + settleTime;
				result.push_back({ std::format("{} {}", cycle, starts[stage].first), starts[stage].second, end });
			}
		}
		return stagesByLength.emplace(events, std::move(result)).first->second;
	}

	std::vector<SerialOrderProfile::Stage> SerialOrderProfile::scoredStages()
	{
		// How many events of each recall are scored, with its X; the demonstration's stages always are.
		static const std::map<std::string, size_t> recalledEvents{ { "demonstration", 0 }, { "recall-start", 1 },
			{ "recall-two", 2 }, { "recall-three", 3 } };
		const std::string part = SolutionConstants::profileScoredStages.value_or("all");
		if (part == "all")
		{
			return stages();
		}
		const size_t events = recalledEvents.at(part);
		std::vector<std::string> scoredInCycle;
		if (events > 0)
		{
			scoredInCycle.emplace_back("X");
		}
		for (size_t event = 0; event < events; ++event)
		{
			scoredInCycle.push_back(std::format("recall {}", serial_order_schedule::sequence[event].label));
		}
		const auto scored = [&scoredInCycle](const Stage& stage) {
			const bool inCycle = std::isdigit(static_cast<unsigned char>(stage.name.front()));
			return !inCycle || std::ranges::find(scoredInCycle, stage.name.substr(stage.name.find(' ') + 1))
				!= scoredInCycle.end();
		};
		std::vector<Stage> result;
		std::ranges::copy_if(stages(), std::back_inserter(result), scored);
		return result;
	}

	void SerialOrderProfile::testPhenotype()
	{
		const ActivationProfile profile = runTrial();
		const ActivationProfile& reference = referenceProfile();

		// Only the moments inside a scored stage are scored; between stages nothing happens.
		parameters.partialFitness.clear();
		std::vector<bool> inStage(profile.times.size(), false);
		double total = 0.0;
		double moments = 0.0;
		for (const Stage& stage : scoredStages())
		{
			double stageTotal = 0.0;
			size_t stageMoments = 0;
			for (size_t sample = firstSampleFrom(profile, stage.start);
				sample < profile.times.size() && profile.times[sample] < stage.end - 1e-9; ++sample)
			{
				stageTotal += momentScore(profile, reference, sample);
				++stageMoments;
				inStage[sample] = true;
			}
			parameters.partialFitness.push_back(stageTotal / static_cast<double>(stageMoments));
			total += stageTotal;
			moments += static_cast<double>(stageMoments);
		}

		if (SolutionConstants::profileCountsOutputActivityOutsideStages.value_or(false))
		{
			// One more term, of the moments outside the scored stages, each counting as much as the output is active
			// then. It counts for at least one moment, filled with a perfect score as far as the output was never
			// active, so a design that stays quiet there scores 1 on it.
			double outsideTotal = 0.0;
			double outsideWeight = 0.0;
			for (size_t sample = 0; sample < profile.times.size(); ++sample)
			{
				if (!inStage[sample])
				{
					const double activity = outputActivity(profile, sample);
					outsideTotal += activity * momentScore(profile, reference, sample);
					outsideWeight += activity;
				}
			}
			const double weight = std::max(outsideWeight, 1.0);
			parameters.partialFitness.push_back((outsideTotal + (weight - outsideWeight)) / weight);
			total += outsideTotal + (weight - outsideWeight);
			moments += weight;
		}
		parameters.fitness = total / moments;
	}

	void SerialOrderProfile::createPhenotypeEnvironment()
	{
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		// The cue X comes first, then the events in the order they are shown; a shorter sequence leaves the last out.
		const auto stimuli = serial_order_schedule::stimuli();
		for (const auto& [stimulusName, stimulusParameters] : std::span(stimuli).first(1 + shownEvents()))
		{
			addTimedGaussianStimulus(stimulusName, inputField, stimulusParameters, dimensions);
		}
	}

	ActivationProfile SerialOrderProfile::runTrial()
	{
		using serial_order_schedule::sampleInterval;

		createPhenotypeEnvironment();
		initSimulation();

		ActivationProfile profile;
		const auto input = getNeuralFieldOrThrow(inputField, "SerialOrderProfile::runTrial");
		const auto output = getNeuralFieldOrThrow(outputField, "SerialOrderProfile::runTrial");
		auto& inputSamples = profile.fields[inputField];
		auto& outputSamples = profile.fields[outputField];
		// Samples fall on whole multiples of the interval, from the start, so every design's line up with the reference's.
		for (long sample = 0; static_cast<double>(sample) * sampleInterval <= serial_order_schedule::endTime + 1e-9; ++sample)
		{
			const double time = static_cast<double>(sample) * sampleInterval;
			while (phenotype.getT() < time - 1e-9)
			{
				runSimulation(1);
			}
			profile.times.push_back(time);
			inputSamples.push_back(input->getComponent("activation"));
			outputSamples.push_back(output->getComponent("activation"));
		}
		return profile;
	}
}
