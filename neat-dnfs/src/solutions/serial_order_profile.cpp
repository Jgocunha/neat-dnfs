#include "solutions/serial_order_profile.h"

#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>

#include "neat_tools/resource_paths.h"
#include "solutions/serial_order_schedule.h"

namespace neat_dnfs
{
	namespace
	{
		const std::string inputField = "nf 1";
		const std::string outputField = "nf 2";

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

	void SerialOrderProfile::testPhenotype()
	{
		throw std::logic_error("SerialOrderProfile has no reward rule yet.");
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
