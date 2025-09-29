#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <string>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#endif

// ANSI Color Codes
#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define BOLD    "\033[1m"
#define DIM     "\033[2m"

// Background colors
#define BG_BLACK   "\033[40m"
#define BG_RED     "\033[41m"
#define BG_GREEN   "\033[42m"
#define BG_YELLOW  "\033[43m"
#define BG_BLUE    "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN    "\033[46m"
#define BG_WHITE   "\033[47m"

class XMRigStyleUI {
private:
    uint64_t total_hashes = 0;
    uint32_t accepted_shares = 0;
    uint32_t rejected_shares = 0;
    uint32_t total_shares = 0;
    double hashrate = 0.0;
    std::chrono::steady_clock::time_point start_time;
    
public:
    XMRigStyleUI() {
        start_time = std::chrono::steady_clock::now();
    }

    void clearScreen() {
        std::cout << "\033[2J\033[H";
    }
    
    void hideCursor() {
        std::cout << "\033[?25l";
    }
    
    void showCursor() {
        std::cout << "\033[?25h";
    }
    
    void gotoxy(int x, int y) {
        std::cout << "\033[" << y << ";" << x << "H";
    }

    std::string formatHashrate(double hashrate) {
        if (hashrate >= 1000000000) {
            return std::to_string(static_cast<int>(hashrate / 1000000000)) + " GH/s";
        } else if (hashrate >= 1000000) {
            return std::to_string(static_cast<int>(hashrate / 1000000)) + " MH/s";
        } else if (hashrate >= 1000) {
            return std::to_string(static_cast<int>(hashrate / 1000)) + " kH/s";
        } else {
            return std::to_string(static_cast<int>(hashrate)) + " H/s";
        }
    }

    std::string getUptime() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        
        int hours = elapsed.count() / 3600;
        int minutes = (elapsed.count() % 3600) / 60;
        int seconds = elapsed.count() % 60;
        
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(2) << hours << ":"
           << std::setfill('0') << std::setw(2) << minutes << ":"
           << std::setfill('0') << std::setw(2) << seconds;
        return ss.str();
    }

    void displayHeader() {
        std::cout << BOLD << CYAN << "╔══════════════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << CYAN << "║                    " << WHITE << "ZION COSMIC HARMONY MINER" << CYAN << "                    ║" << RESET << std::endl;
        std::cout << BOLD << CYAN << "║                      " << YELLOW << "XMRig-Style Interface" << CYAN << "                     ║" << RESET << std::endl;
        std::cout << BOLD << CYAN << "╚══════════════════════════════════════════════════════════════╝" << RESET << std::endl;
    }

    void displayStats() {
        std::cout << std::endl;
        
        // Mining stats box
        std::cout << CYAN << "┌─ " << WHITE << BOLD << "MINING STATISTICS" << RESET << CYAN << " ──────────────────────────────────┐" << RESET << std::endl;
        
        // Hashrate
        std::cout << CYAN << "│" << RESET << " Hashrate:     " << GREEN << BOLD << std::setw(15) << formatHashrate(hashrate) << RESET;
        std::cout << "                    " << CYAN << "│" << RESET << std::endl;
        
        // Uptime
        std::cout << CYAN << "│" << RESET << " Uptime:       " << YELLOW << BOLD << std::setw(15) << getUptime() << RESET;
        std::cout << "                    " << CYAN << "│" << RESET << std::endl;
        
        // Total hashes
        std::cout << CYAN << "│" << RESET << " Total Hashes: " << WHITE << BOLD << std::setw(15) << total_hashes << RESET;
        std::cout << "                    " << CYAN << "│" << RESET << std::endl;
        
        std::cout << CYAN << "└───────────────────────────────────────────────────────────────┘" << RESET << std::endl;
        
        std::cout << std::endl;
        
        // Shares stats box
        std::cout << CYAN << "┌─ " << WHITE << BOLD << "SHARE STATISTICS" << RESET << CYAN << " ───────────────────────────────────┐" << RESET << std::endl;
        
        // Accepted shares
        std::cout << CYAN << "│" << RESET << " Accepted:     " << GREEN << BOLD << std::setw(10) << accepted_shares << RESET;
        double accept_rate = total_shares > 0 ? (double(accepted_shares) / total_shares * 100.0) : 0.0;
        std::cout << " (" << GREEN << std::fixed << std::setprecision(1) << accept_rate << "%" << RESET << ")";
        std::cout << "       " << CYAN << "│" << RESET << std::endl;
        
        // Rejected shares
        std::cout << CYAN << "│" << RESET << " Rejected:     " << RED << BOLD << std::setw(10) << rejected_shares << RESET;
        double reject_rate = total_shares > 0 ? (double(rejected_shares) / total_shares * 100.0) : 0.0;
        std::cout << " (" << RED << std::fixed << std::setprecision(1) << reject_rate << "%" << RESET << ")";
        std::cout << "       " << CYAN << "│" << RESET << std::endl;
        
        // Total shares
        std::cout << CYAN << "│" << RESET << " Total:        " << WHITE << BOLD << std::setw(10) << total_shares << RESET;
        std::cout << "                    " << CYAN << "│" << RESET << std::endl;
        
        std::cout << CYAN << "└───────────────────────────────────────────────────────────────┘" << RESET << std::endl;
    }

    void displayActivity() {
        std::cout << std::endl;
        std::cout << CYAN << "┌─ " << WHITE << BOLD << "MINING ACTIVITY" << RESET << CYAN << " ────────────────────────────────────┐" << RESET << std::endl;
        
        // Show last few activities
        if (total_shares > 0) {
            std::cout << CYAN << "│" << RESET << " " << GREEN << "✓" << RESET << " Share accepted   ";
            std::cout << DIM << "[" << getUptime() << "]" << RESET << "                " << CYAN << "│" << RESET << std::endl;
        }
        
        if (hashrate > 0) {
            std::cout << CYAN << "│" << RESET << " " << BLUE << "⚡" << RESET << " Mining at " << GREEN << formatHashrate(hashrate) << RESET;
            std::cout << "                           " << CYAN << "│" << RESET << std::endl;
        }
        
        std::cout << CYAN << "│" << RESET << " " << YELLOW << "◆" << RESET << " ZION Cosmic Harmony Algorithm Active              " << CYAN << "│" << RESET << std::endl;
        
        std::cout << CYAN << "└───────────────────────────────────────────────────────────────┘" << RESET << std::endl;
    }

    void displayFooter() {
        std::cout << std::endl;
        std::cout << DIM << "Press Ctrl+C to stop mining..." << RESET << std::endl;
        std::cout << DIM << "ZION v1.0.0 - Built for victory and liberation" << RESET << std::endl;
    }

    void shareAccepted() {
        accepted_shares++;
        total_shares++;
        
        // Clear screen and redraw
        clearScreen();
        displayAll();
        
        // Show notification
        std::cout << std::endl;
        std::cout << GREEN << BOLD << "🎉 SHARE ACCEPTED! 🎉" << RESET << std::endl;
        std::cout << GREEN << "New share found and accepted by pool!" << RESET << std::endl;
        
        // Beep sound (if supported)
        std::cout << "\a";
        std::flush(std::cout);
    }

    void shareRejected() {
        rejected_shares++;
        total_shares++;
        
        // Clear screen and redraw
        clearScreen();
        displayAll();
        
        // Show notification
        std::cout << std::endl;
        std::cout << RED << BOLD << "❌ SHARE REJECTED ❌" << RESET << std::endl;
        std::cout << RED << "Share was rejected by pool" << RESET << std::endl;
    }

    void blockFound() {
        // Clear screen and redraw
        clearScreen();
        displayAll();
        
        // Show MASSIVE notification
        std::cout << std::endl;
        std::cout << BG_GREEN << BLACK << BOLD << "████████████████████████████████████████████████████████████████" << RESET << std::endl;
        std::cout << BG_GREEN << BLACK << BOLD << "█                                                              █" << RESET << std::endl;
        std::cout << BG_GREEN << BLACK << BOLD << "█                     🏆 BLOCK FOUND! 🏆                      █" << RESET << std::endl;
        std::cout << BG_GREEN << BLACK << BOLD << "█                                                              █" << RESET << std::endl;
        std::cout << BG_GREEN << BLACK << BOLD << "████████████████████████████████████████████████████████████████" << RESET << std::endl;
        std::cout << GREEN << BOLD << "Congratulations! You found a block!" << RESET << std::endl;
        std::cout << YELLOW << "ZION Cosmic Harmony brings victory!" << RESET << std::endl;
        
        // Multiple beeps for block found
        for (int i = 0; i < 5; i++) {
            std::cout << "\a";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::flush(std::cout);
    }

    void updateHashrate(double new_hashrate) {
        hashrate = new_hashrate;
        total_hashes += static_cast<uint64_t>(new_hashrate);
    }

    void displayAll() {
        displayHeader();
        displayStats();
        displayActivity();
        displayFooter();
    }
};

// Simple mining simulation
class ZionMiner {
private:
    XMRigStyleUI ui;
    bool running = false;
    
public:
    void start() {
        running = true;
        ui.hideCursor();
        
        while (running) {
            ui.clearScreen();
            
            // Simulate mining activity
            double simulated_hashrate = 1500000 + (rand() % 500000); // 1.5-2 MH/s
            ui.updateHashrate(simulated_hashrate);
            
            ui.displayAll();
            
            // Simulate finding shares occasionally
            if (rand() % 100 < 5) { // 5% chance per update
                std::this_thread::sleep_for(std::chrono::seconds(1));
                ui.shareAccepted();
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            
            // Very rare block find
            if (rand() % 10000 < 1) { // 0.01% chance
                std::this_thread::sleep_for(std::chrono::seconds(1));
                ui.blockFound();
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
        
        ui.showCursor();
    }
    
    void stop() {
        running = false;
    }
};

int main() {
    std::cout << BOLD << CYAN << "Starting ZION Cosmic Harmony Miner..." << RESET << std::endl;
    std::cout << YELLOW << "Initializing XMRig-style interface..." << RESET << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    ZionMiner miner;
    
    try {
        miner.start();
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
        return 1;
    }
    
    return 0;
}