#include "zion.h"
#include "transaction.h"
#include "randomx_wrapper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>

namespace zion {

class ZionWallet {
public:
    ZionWallet() : balance_(0), keys_loaded_(false) {}
    
    bool initialize() {
        std::cout << "Inicializace ZION wallet..." << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "ZION Wallet - CLI rozhraní" << std::endl;
        std::cout << "Zadejte 'help' pro nápovědu" << std::endl;
        
        std::string command;
        while (true) {
            std::cout << "\nzion-wallet> ";
            std::getline(std::cin, command);
            
            if (command.empty()) continue;
            
            std::istringstream iss(command);
            std::string cmd;
            iss >> cmd;
            
            if (cmd == "help") {
                show_help();
            }
            else if (cmd == "generate") {
                generate_new_address();
            }
            else if (cmd == "load") {
                std::string filename;
                iss >> filename;
                load_wallet(filename);
            }
            else if (cmd == "save") {
                std::string filename;
                iss >> filename;
                save_wallet(filename);
            }
            else if (cmd == "balance") {
                show_balance();
            }
            else if (cmd == "address") {
                show_address();
            }
            else if (cmd == "send") {
                std::string recipient_str, amount_str;
                iss >> recipient_str >> amount_str;
                send_transaction(recipient_str, amount_str);
            }
            else if (cmd == "broadcasthex") {
                std::string txhex;
                iss >> txhex;
                broadcast_tx_hex(txhex);
            }
            else if (cmd == "history") {
                show_transaction_history();
            }
            else if (cmd == "exit" || cmd == "quit") {
                break;
            }
            else {
                std::cout << "Neznámý příkaz: " << cmd << std::endl;
                std::cout << "Zadejte 'help' pro nápovědu" << std::endl;
            }
        }
        
        std::cout << "Ukončuji wallet..." << std::endl;
    }
    
private:
    PrivateKey private_key_;
    PublicKey public_key_;
    uint64_t balance_;
    std::vector<std::string> transaction_history_;
    bool keys_loaded_;
    
    void show_help() {
        std::cout << "Dostupné příkazy:" << std::endl;
        std::cout << "  help                    - Zobrazit tuto nápovědu" << std::endl;
        std::cout << "  generate                - Vygenerovat novou adresu" << std::endl;
        std::cout << "  load <soubor>          - Načíst wallet ze souboru" << std::endl;
        std::cout << "  save <soubor>          - Uložit wallet do souboru" << std::endl;
        std::cout << "  balance                - Zobrazit aktuální zůstatek" << std::endl;
        std::cout << "  address                - Zobrazit adresu peněženky" << std::endl;
        std::cout << "  send <adresa> <částka> - Poslat ZION na adresu" << std::endl;
        std::cout << "  history                - Zobrazit historii transakcí" << std::endl;
        std::cout << "  exit/quit              - Ukončit wallet" << std::endl;
    }
    
    void generate_new_address() {
        std::cout << "Generuji novou adresu..." << std::endl;
        
        auto [private_key, public_key] = generate_keypair();
        private_key_ = private_key;
        public_key_ = public_key;
        keys_loaded_ = true;
        
        std::cout << "✓ Nová adresa vygenerována:" << std::endl;
        show_address();
        
        std::cout << "DŮLEŽITÉ: Uložte si peněženku pomocí příkazu 'save <soubor>'" << std::endl;
    }
    
    void load_wallet(const std::string& filename) {
        if (filename.empty()) {
            std::cout << "Chyba: Zadejte název souboru" << std::endl;
            return;
        }
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Chyba: Nelze otevřít soubor " << filename << std::endl;
            return;
        }
        
        // Simple file format: private_key (32 bytes) + public_key (32 bytes)
        file.read(reinterpret_cast<char*>(private_key_.data()), 32);
        file.read(reinterpret_cast<char*>(public_key_.data()), 32);
        
        if (file.gcount() == 32) {
            keys_loaded_ = true;
            std::cout << "✓ Wallet načten ze souboru " << filename << std::endl;
            show_address();
        } else {
            std::cout << "Chyba: Neplatný formát souboru" << std::endl;
            keys_loaded_ = false;
        }
        
        file.close();
    }
    
    void save_wallet(const std::string& filename) {
        if (filename.empty()) {
            std::cout << "Chyba: Zadejte název souboru" << std::endl;
            return;
        }
        
        if (!keys_loaded_) {
            std::cout << "Chyba: Nejprve vygenerujte nebo načtěte adresu" << std::endl;
            return;
        }
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Chyba: Nelze vytvořit soubor " << filename << std::endl;
            return;
        }
        
        file.write(reinterpret_cast<const char*>(private_key_.data()), 32);
        file.write(reinterpret_cast<const char*>(public_key_.data()), 32);
        
        file.close();
        
        std::cout << "✓ Wallet uložen do souboru " << filename << std::endl;
        std::cout << "VAROVÁNÍ: Uchovejte tento soubor v bezpečí!" << std::endl;
    }
    
    void show_balance() {
        if (!keys_loaded_) {
            std::cout << "Nejprve vygenerujte nebo načtěte adresu" << std::endl;
            return;
        }
        
        // In a real implementation, we would query the blockchain for UTXO
        std::cout << "Aktuální zůstatek: " << format_amount(balance_) << " ZION" << std::endl;
        std::cout << "Poznámka: V této demo verzi je zůstatek pouze simulovaný" << std::endl;
    }
    
    void show_address() {
        if (!keys_loaded_) {
            std::cout << "Nejprve vygenerujte nebo načtěte adresu" << std::endl;
            return;
        }
        
        std::cout << "Vaše ZION adresa:" << std::endl;
        for (const auto& byte : public_key_) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
        }
        std::cout << std::dec << std::endl;
    }
    
    void send_transaction(const std::string& recipient_str, const std::string& amount_str) {
        if (!keys_loaded_) {
            std::cout << "Nejprve vygenerujte nebo načtěte adresu" << std::endl;
            return;
        }
        
        if (recipient_str.empty() || amount_str.empty()) {
            std::cout << "Použití: send <adresa> <částka>" << std::endl;
            return;
        }
        
        // Parse recipient address
        if (recipient_str.length() != 64) {
            std::cout << "Chyba: Neplatná adresa (musí být 64 hex znaků)" << std::endl;
            return;
        }
        
        PublicKey recipient;
        try {
            for (size_t i = 0; i < 32; ++i) {
                std::string byte_str = recipient_str.substr(i * 2, 2);
                recipient[i] = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
            }
        } catch (const std::exception& e) {
            std::cout << "Chyba: Neplatná adresa (neplatné hex znaky)" << std::endl;
            return;
        }
        
        // Parse amount
        uint64_t amount;
        try {
            double amount_double = std::stod(amount_str);
            amount = static_cast<uint64_t>(amount_double * 1000000); // Convert to micro-ZION
        } catch (const std::exception& e) {
            std::cout << "Chyba: Neplatná částka" << std::endl;
            return;
        }
        
        if (amount == 0) {
            std::cout << "Chyba: Částka musí být větší než 0" << std::endl;
            return;
        }
        
        if (amount > balance_) {
            std::cout << "Chyba: Nedostatečný zůstatek" << std::endl;
            return;
        }
        
        // Create transaction (simplified)
        std::cout << "Vytvářím transakci..." << std::endl;
        
        Transaction tx;
        
        // In a real implementation, we would:
        // 1. Find suitable UTXOs from the blockchain
        // 2. Create inputs referencing those UTXOs
        // 3. Sign the transaction
        // 4. Broadcast to the network
        
        // For demo purposes, just show what would happen
        std::cout << "✓ Transakce vytvořena (demo):" << std::endl;
        std::cout << "  Příjemce: " << recipient_str << std::endl;
        std::cout << "  Částka: " << format_amount(amount) << " ZION" << std::endl;
        std::cout << "  Poplatek: " << format_amount(1000) << " ZION" << std::endl;
        
        // Update balance (demo)
        balance_ -= (amount + 1000); // amount + fee
        
        // Add to history
        std::stringstream history_entry;
        history_entry << "ODCHOZÍ: " << format_amount(amount) << " ZION -> " 
                     << recipient_str.substr(0, 8) << "...";
        transaction_history_.push_back(history_entry.str());
        
        std::cout << "Poznámka: V této demo verzi se transakce nevysílá do sítě" << std::endl;
    }
    
    void broadcast_tx_hex(const std::string& txhex) {
        if (txhex.empty()) { std::cout << "Použití: broadcasthex <tx_hex>" << std::endl; return; }
        // Submit TX via local pool port as SUBMITTX (placeholder server returns ERR now)
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { std::cout << "Chyba: nelze otevřít socket" << std::endl; return; }
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(3333); inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { std::cout << "Chyba: nelze se připojit na 127.0.0.1:3333" << std::endl; ::close(fd); return; }
        std::string line = std::string("SUBMITTX ") + txhex + "\n";
        send(fd, line.data(), line.size(), 0);
        std::string resp; char ch; while (recv(fd, &ch, 1, 0) > 0) { if (ch=='\n') break; resp.push_back(ch); }
        std::cout << "Odpověď: " << resp << std::endl;
        ::close(fd);
    }

    void show_transaction_history() {
        if (transaction_history_.empty()) {
            std::cout << "Žádné transakce v historii" << std::endl;
            return;
        }
        
        std::cout << "Historie transakcí:" << std::endl;
        for (size_t i = 0; i < transaction_history_.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << transaction_history_[i] << std::endl;
        }
    }
    
    std::string format_amount(uint64_t amount) {
        double zion_amount = static_cast<double>(amount) / 1000000.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << zion_amount;
        return oss.str();
    }
};

} // namespace zion

int main(int argc, char* argv[]) {
    std::cout << "ZION Cryptocurrency Wallet v"
              << static_cast<int>(zion::ZION_VERSION_MAJOR) << "."
              << static_cast<int>(zion::ZION_VERSION_MINOR) << "."
              << static_cast<int>(zion::ZION_VERSION_PATCH) << std::endl;
    
    zion::ZionWallet wallet;
    
    if (!wallet.initialize()) {
        std::cerr << "Chyba při inicializaci wallet!" << std::endl;
        return 1;
    }
    
    wallet.run();
    
    return 0;
}
