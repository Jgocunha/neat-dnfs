//This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include "neat_tools/logger.h"
#include <format>
#include <mutex>

namespace neat_dnfs
{
    namespace tools
    {
        namespace logger
        {
        	LogLevel Logger::minLogLevel = LogLevel::DEBUG; 

            Logger::Logger(LogLevel level, LogOutputMode mode)
                : logLevel(level), outputMode(mode)
            {}

            void Logger::log(const std::string& message) const
            {
                if (logLevel < Logger::minLogLevel)
                    return;

                const auto now = std::chrono::system_clock::now();
                const auto in_time_t = std::chrono::system_clock::to_time_t(now);

                std::tm buf;
#ifdef _WIN32
                if (localtime_s(&buf, &in_time_t))
                    throw std::runtime_error("Failed to get current time");
#else
                if (!localtime_r(&in_time_t, &buf))
                    throw std::runtime_error("Failed to get current time");
#endif

                const std::string levelStr = getLogLevelText(logLevel);
                const std::string prefixStr = std::format("<neat-dnfs> {}", levelStr);
                std::ostringstream timeStream;
                timeStream << std::put_time(&buf, "%Y-%m-%d %X");
                const std::string timeStr = timeStream.str();
                std::string colorCode;

                const ImVec4 color = getLogLevelColorCodeGui(logLevel);
                switch (outputMode)
                {
                case LogOutputMode::ALL:
                    colorCode = getLogLevelColorCodeCmd(logLevel);
                    log_cmd(std::format("{}[{}] {} {}", colorCode, timeStr, prefixStr, message));
                    log_ui(color, std::format("[{}] {}  {}", timeStr, prefixStr, message));
                    break;
                case LogOutputMode::CONSOLE:
                    colorCode = getLogLevelColorCodeCmd(logLevel);
                    log_cmd(std::format("{}[{}] {} {}", colorCode, timeStr, prefixStr, message));
                    break;
                case LogOutputMode::GUI:
                    log_ui(color, std::format("[{}] {}  {}", timeStr, prefixStr, message));
                    break;
                default:
                    break;
                }
            }

            void Logger::log_cmd(const std::string& message)
            {
                static std::mutex coutMutex;
                std::scoped_lock lock(coutMutex);
                std::cout << message << "\033[0m\n";
            }

            void Logger::log_ui(ImVec4 color, const std::string& message)
            {
                imgui_kit::LogWindow::addLog(color, "%s", message.c_str());
            }

            void log(LogLevel level, const std::string& message, LogOutputMode mode)
            {
#ifndef _DEBUG
                if (level == LogLevel::DEBUG)
                    return;
#endif

                Logger(level, mode).log(message);
            }

            std::string Logger::getLogLevelColorCodeCmd(LogLevel level)
            {
                switch (level)
                {
                case DEBUG:     return "\033[92m"; // Green
                case INFO:      return"\033[0m";
                case WARNING:   return"\033[93m";  // Yellow
                case ERROR:
                case FATAL:     return"\033[91m";  // Red
                default:        return "\033[0m";
                }
            }

            ImVec4 Logger::getLogLevelColorCodeGui(LogLevel level)
            {
                ImVec4 currentTextColor = imgui_kit::colours::White;
                if (ImGui::GetCurrentContext())
                {
                    const ImGuiStyle& style = ImGui::GetStyle();
                    currentTextColor = style.Colors[ImGuiCol_Text];
                }

                switch (level)
                {
                case DEBUG:     return imgui_kit::colours::Green;
                case INFO:      return currentTextColor;
                case WARNING:   return imgui_kit::colours::Yellow;
                case ERROR:
                case FATAL:     return imgui_kit::colours::Red;
                default:        return currentTextColor;
                }
            }

            std::string Logger::getLogLevelText(LogLevel level)
            {
                switch (level)
                {
                case DEBUG: return      "DEBUG   ";
                case INFO: return       "INFO    ";
                case WARNING: return    "WARNING ";
                case ERROR: return      "ERROR   ";
                case FATAL: return      "FATAL   ";
                default: return         "UNKNOWN ";
                }
            }
        }
    }
}

