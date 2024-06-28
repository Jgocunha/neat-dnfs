#include "user_interface/main_window.h"


namespace neat_dnfs
{
	MainWindow::MainWindow()
		: population{ nullptr }
	{
	}

	void MainWindow::render()
	{
        if(ImGui::Begin("Main Window"))
        {
	        renderPopulationInfo();
			renderSolutionInfo();
            renderPopulationMethods();
            renderSolutionMethods();
		}
        ImGui::End();
	}

    MainWindow::~MainWindow()
	{
		if (evolveThread && evolveThread->joinable())
			evolveThread->join();
        if (simulationThread && simulationThread->joinable())
			simulationThread->join();
	}

	void MainWindow::renderPopulationInfo()
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

        static double targetFitness = 0.8f;
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

	void MainWindow::renderSolutionInfo()
	{
		static constexpr int columnWidth = 250;
        static constexpr int inputWidth = 150;

        ImGui::Columns(2, "SolutionColumns", false);

        ImGui::SetColumnWidth(0, columnWidth);

        static int numberOfInputFields = 2;
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

    void MainWindow::renderPopulationMethods()
    {
        static ImVec2 buttonSize = ImVec2(120, 40);
        static bool isInitialized = false;

        if (ImGui::Button("Initialize", buttonSize))
        {
            ActionSimulationLayerSolution solution(solutionParameters);
            population = std::make_shared<Population>(populationParameters,
                std::make_shared<ActionSimulationLayerSolution>(solution));
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
            if (evolveThread && evolveThread->joinable())
                evolveThread->join();
            if (simulationThread && simulationThread->joinable())
                simulationThread->join();

        	evolveThread = std::make_shared<std::thread>(&Population::evolve, population);
        }

        if (!isInitialized) 
            ImGui::EndDisabled(); 

        ImGui::Spacing();
    }

    void MainWindow::renderSolutionMethods()
	{
		if (!population)
			return;
		if (!population->getBestSolution())
            return;

        static ImVec2 buttonSize = ImVec2(180, 40);
        static bool showBestSolution = false;

        if (ImGui::Button("Show best solution", buttonSize))
        {
            if (evolveThread && evolveThread->joinable())
                evolveThread->join();
            if (simulationThread && simulationThread->joinable())
                simulationThread->join();

        	bestSolution = population->getBestSolution();
            bestSolution->buildPhenotype();
        	auto phenotype = bestSolution->getPhenotype();
            phenotype.init();
        	simulation = std::make_shared<dnf_composer::Simulation>(phenotype);
            simulationWindow = std::make_shared<dnf_composer::user_interface::SimulationWindow>(simulation);
            elementWindow = std::make_shared<dnf_composer::user_interface::ElementWindow>(simulation);
            fieldMetricsWindow = std::make_shared<dnf_composer::user_interface::FieldMetricsWindow>(simulation);

            const char* filename = "NodeEditor.json";
            if (std::remove(filename) == 0)
                std::cout << "File deleted successfully" << std::endl;
            else
                std::perror("File deletion failed");

            nodeGraphWindow = std::make_shared<dnf_composer::user_interface::NodeGraphWindow>(simulation);

        	const auto visualization = createVisualization(simulation);
            for (const auto& element : simulation->getElements())
            {
                if (element->getLabel() == dnf_composer::element::ElementLabel::NEURAL_FIELD)
                    visualization->addPlottingData(element->getUniqueName(), "activation");
            }

            const auto plot = std::make_shared<dnf_composer::user_interface::PlotWindow>(visualization);
            plotWindows.push_back(plot);

            showBestSolution = true;
            simulationThread = std::make_shared<std::thread>(&MainWindow::renderShowBestSolution, this);
        }
        if (showBestSolution)
        {
            simulationWindow->render();
            elementWindow->render();
            fieldMetricsWindow->render();
            nodeGraphWindow->render();
            for (const auto& plot : plotWindows)
                plot->render();
        }
	}

    void MainWindow::renderShowBestSolution()
    {
        using namespace dnf_composer::element;
        const ElementCommonParameters elementCommonParameters_1{ "gauss stimulus 1", {100, 1.0} };
        const ElementCommonParameters elementCommonParameters_2{ "gauss stimulus 2", {100, 1.0} };

        const GaussStimulusParameters gaussStimulusParameters{ 5, 15, 25, false, false };
        const auto gaussStimulus_1 = std::make_shared<GaussStimulus>(elementCommonParameters_1, gaussStimulusParameters);
        const auto gaussStimulus_2 = std::make_shared<GaussStimulus>(elementCommonParameters_2, gaussStimulusParameters);

        simulation->addElement(gaussStimulus_1);
        simulation->addElement(gaussStimulus_2);

        simulation->createInteraction("gauss stimulus 1", "output", "nf 1");
        simulation->createInteraction("gauss stimulus 2", "output", "nf 2");

        // re-declare interactions for all elements
        for (const auto& element : simulation->getElements())
        {
            const auto inputs = element->getInputs();
            for(const auto& input : inputs)
            {
                simulation->createInteraction(input->getUniqueName(), "output", element->getUniqueName());
			}
		}

        simulation->init();
        do
        {
            try
            {
				simulation->step();
        		Sleep(1);
            }
            catch (const dnf_composer::Exception& ex)
            {
                log(tools::logger::LogLevel::FATAL, "Exception caught: " + std::string(ex.what()) + ".");
                break;
            }
        	catch (const std::exception& e)
			{
				std::cerr << e.what() << std::endl;
				break;
			}
            catch (...)
            {
                std::cerr << "Unknown exception" << std::endl;
                break;
            }
        } while (true);
    }
}
