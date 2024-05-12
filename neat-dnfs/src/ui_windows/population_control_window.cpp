#include "ui_windows/population_control_window.h"

namespace neat_dnfs
{
	PopulationControlWindow::PopulationControlWindow(const std::shared_ptr<Population>& population)
		: population(population), populationParameters(), solutionParameters()
	{
	}

	void PopulationControlWindow::render()
	{
		if(ImGui::Begin("Population control"))
		{
			renderPopulationInfo();
			renderSolutionInfo();
            renderPopulationMethods();
            renderShowBestSolution();
		}
		ImGui::End();
	}

    void PopulationControlWindow::renderPopulationInfo()
    {
        static constexpr int columnWidth = 250;
        static constexpr int inputWidth = 150;

        ImGui::Columns(2, "PopulationColumns", false);

        ImGui::SetColumnWidth(0, columnWidth);

        static int populationSize = 100;
        ImGui::Text("Population size");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(inputWidth);
        ImGui::InputInt("##PopulationSize", &populationSize, 10, 100);
        ImGui::NextColumn();

        static int numberOfGenerations = 1000;
        ImGui::Text("Number of generations");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(inputWidth);
        ImGui::InputInt("##NumberGenerations", &numberOfGenerations, 10, 100);
        ImGui::NextColumn();

        static double targetFitness = 0.95f;
        ImGui::Text("Target fitness");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(inputWidth);
        ImGui::InputDouble("##TargetFitness", &targetFitness, 0.05, 0.1, "%.2f");
        ImGui::NextColumn();

        ImGui::Columns(1);

        populationParameters.size = static_cast<uint16_t>(populationSize);
        populationParameters.numGenerations = static_cast<uint16_t>(numberOfGenerations);
        populationParameters.targetFitness = targetFitness;
    }

    void PopulationControlWindow::renderSolutionInfo()
    {
        static constexpr int columnWidth = 250;
        static constexpr int inputWidth = 150;

        ImGui::Columns(2, "SolutionColumns", false);

        ImGui::SetColumnWidth(0, columnWidth);

        static int numberOfInputFields = 3;
        ImGui::Text("Number of input fields");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(inputWidth);
        ImGui::InputInt("##NumberOfInputFields", &numberOfInputFields, 1, 2);
        ImGui::NextColumn();

        static int numberOfOutputFields = 1;
        ImGui::Text("Number of output fields");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(inputWidth);
        ImGui::InputInt("##NumberOfOutputFields", &numberOfOutputFields, 1, 2);
        ImGui::NextColumn();

        ImGui::Columns(1);

        solutionParameters.numInputGenes = static_cast<uint16_t>(numberOfInputFields);
        solutionParameters.numOutputGenes = static_cast<uint16_t>(numberOfOutputFields);
        solutionParameters.numHiddenGenes = 0;
    }

    void PopulationControlWindow::renderPopulationMethods()
    {
        static ImVec2 buttonSize = ImVec2(120, 40);
        static bool isInitialized = false;

        if (ImGui::Button("Initialize", buttonSize))
        {
            TemplateSolution solution(solutionParameters);
            population = std::make_shared<Population>(populationParameters,
                std::make_shared<TemplateSolution>(solution));
            population->initialize();
            isInitialized = true;
        }
        ImGui::SameLine(); 

        // Small spacer between buttons
        ImGui::Dummy(ImVec2(5, 0));
        ImGui::SameLine();

        if (!isInitialized) 
            ImGui::BeginDisabled(); 
        
        if (ImGui::Button("Evolve", buttonSize))
        {
            if (evolveThread && evolveThread->joinable()) {
                evolveThread->join();  // Ensure the previous thread is finished before starting a new one
            }
            // Launch the evolve function in a new thread
            evolveThread = std::make_shared<std::thread>(&Population::evolve, population);
        }

        if (!isInitialized) 
            ImGui::EndDisabled(); 

        ImGui::Spacing();
    }

    void PopulationControlWindow::renderShowBestSolution()
    {
        static bool showBestSolution = false;
	    if (ImGui::Button("Show best solution"))
		{
			const auto bestSolution = population->getBestSolution();
            const auto phenotype = bestSolution->getPhenotype();
            const auto simulation = std::make_shared<dnf_composer::Simulation>(phenotype);
            simulationWindow = std::make_shared<dnf_composer::user_interface::SimulationWindow>(simulation);
            elementWindow = std::make_shared<dnf_composer::user_interface::ElementWindow>(simulation);
            centroidMonitoringWindow = std::make_shared<dnf_composer::user_interface::FieldMetricsWindow>(simulation);

            const auto visualization = createVisualization(simulation);
            visualization->addPlottingData("nf 1", "activation");
            visualization->addPlottingData("nf 2", "activation");
            visualization->addPlottingData("nf 3", "activation");
            visualization->addPlottingData("nf 4", "activation");

            const auto plot = std::make_shared<dnf_composer::user_interface::PlotWindow>(visualization);
            plotWindows.push_back(plot);

            showBestSolution = true;
		}

        if (showBestSolution)
        {
            simulationWindow->render();
            elementWindow->render();
            centroidMonitoringWindow->render();

            for (const auto& plot : plotWindows)
			{
				plot->render();
			}
        }
    }

    PopulationControlWindow::~PopulationControlWindow()
	{
		if (evolveThread && evolveThread->joinable())
			evolveThread->join();
	}
}