#include "solutions/serial_order_profile.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numeric>
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
		// After a pulse ends, the input, a detection field with tau 10, is back at rest within five time constants.
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

		std::filesystem::path referenceProfilePath()
		{
			return paths::resourceRoot() / "config" / "references" / "serial-order-profile.json";
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
		static const ActivationProfile reference = [] {
			std::ifstream file(referenceProfilePath());
			if (!file)
			{
				throw std::runtime_error(std::format("Cannot read the reference profile {}.",
					referenceProfilePath().generic_string()));
			}
			return ActivationProfile::fromJson(nlohmann::json::parse(file));
		}();
		return reference;
	}

	const std::vector<SerialOrderProfile::Stage>& SerialOrderProfile::stages()
	{
		static const std::vector<Stage> trialStages = [] {
			using namespace serial_order_schedule;
			std::vector<Stage> result{ { "trigger", cueTimes.front(), sequence.front().onset } };
			for (size_t event = 0; event < sequence.size(); ++event)
			{
				const double end = event + 1 < sequence.size() ? sequence[event + 1].onset
					: sequence[event].onset + pulseDuration + settleTime;
				result.push_back({ std::format("event {}", sequence[event].label), sequence[event].onset, end });
			}
			result.push_back({ "wait", result.back().end, cueTimes[1] });

			const ActivationProfile& reference = referenceProfile();
			for (size_t cycle = 1; cycle < cueTimes.size(); ++cycle)
			{
				std::vector<std::pair<std::string, double>> starts{ { "X", cueTimes[cycle] } };
				for (const auto& event : sequence)
				{
					starts.emplace_back(std::format("recall {}", event.label),
						firstPeakAt(reference, event.position, starts.back().second));
				}
				starts.emplace_back("rest", firstQuietFrom(reference, starts.back().second));
				const double cycleEnd = cycle + 1 < cueTimes.size() ? cueTimes[cycle + 1] : endTime;
				for (size_t stage = 0; stage < starts.size(); ++stage)
				{
					const double end = stage + 1 < starts.size() ? starts[stage + 1].second : cycleEnd;
					result.push_back({ std::format("{} {}", cycle, starts[stage].first), starts[stage].second, end });
				}
			}
			return result;
		}();
		return trialStages;
	}

	void SerialOrderProfile::testPhenotype()
	{
		const ActivationProfile profile = runTrial();
		const ActivationProfile& reference = referenceProfile();

		// Each moment scores how closely each field matches the reference's at that moment, averaged over the fields.
		std::vector<double> momentScores(profile.times.size(), 0.0);
		for (const auto& [fieldName, samples] : profile.fields)
		{
			const auto& referenceSamples = reference.fields.at(fieldName);
			for (size_t sample = 0; sample < samples.size(); ++sample)
			{
				const double agreement = tools::utils::concordance(samples[sample], referenceSamples[sample], flatTolerance);
				momentScores[sample] += (1.0 + agreement) / 2.0 / static_cast<double>(profile.fields.size());
			}
		}

		parameters.partialFitness.clear();
		const auto& trialStages = stages();
		size_t sample = 0;
		for (size_t stage = 0; stage < trialStages.size(); ++stage)
		{
			const bool lastStage = stage + 1 == trialStages.size();
			double total = 0.0;
			size_t count = 0;
			for (; sample < momentScores.size() && (lastStage || profile.times[sample] < trialStages[stage].end - 1e-9);
				++sample)
			{
				total += momentScores[sample];
				++count;
			}
			parameters.partialFitness.push_back(total / static_cast<double>(count));
		}
		parameters.fitness = std::accumulate(momentScores.begin(), momentScores.end(), 0.0)
			/ static_cast<double>(momentScores.size());
	}

	void SerialOrderProfile::createPhenotypeEnvironment()
	{
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		for (const auto& [stimulusName, stimulusParameters] : serial_order_schedule::stimuli())
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
