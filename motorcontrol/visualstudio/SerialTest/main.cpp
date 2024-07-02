#include <iostream>
#include <stdio.h>
#include <windows.h>
#include <time.h>

extern "C" {
#include "mcp_config.h"
#include "aspep.h"
}




// Function to initialize the serial port
HANDLE init_serial(const wchar_t* port_name, int baud_rate) {
    HANDLE hSerial = CreateFile(
        port_name,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hSerial == INVALID_HANDLE_VALUE) {
        if (GetLastError() == ERROR_FILE_NOT_FOUND) {
            wprintf(L"Error: Serial port %s not found.\n", port_name);
        }
        else {
            wprintf(L"Error: Unable to open serial port %s.\n", port_name);
        }
        return INVALID_HANDLE_VALUE;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

    if (!GetCommState(hSerial, &dcbSerialParams)) {
        wprintf(L"Error: Getting state of serial port %s.\n", port_name);
        CloseHandle(hSerial);
        return INVALID_HANDLE_VALUE;
    }

    dcbSerialParams.BaudRate = baud_rate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;

    if (!SetCommState(hSerial, &dcbSerialParams)) {
        wprintf(L"Error: Setting state of serial port %s.\n", port_name);
        CloseHandle(hSerial);
        return INVALID_HANDLE_VALUE;
    }

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;

    if (!SetCommTimeouts(hSerial, &timeouts)) {
        wprintf(L"Error: Setting timeouts for serial port %s.\n", port_name);
        CloseHandle(hSerial);
        return INVALID_HANDLE_VALUE;
    }

    return hSerial;
}

// Function to send a message
int send_message(HANDLE hSerial, const char* message) {
    DWORD bytes_written;
    if (!WriteFile(hSerial, message, strlen(message), &bytes_written, NULL)) {
        printf("Error: Writing to serial port.\n");
        return -1;
    }
    return 0;
}

void sleep_ms(int milliseconds) {
    Sleep(milliseconds);
}

int main() {
    const wchar_t* serial_port = L"\\\\.\\COM4"; // Update this with your serial port
    int baud_rate = 921600; // Update this based on your configuration
    int N = 3; // Number of times to send the message
    int delay_ms = 1000; // Delay between messages in milliseconds

    HANDLE hSerial = init_serial(serial_port, baud_rate);
    if (hSerial == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Failed to open serial port\n");
        return EXIT_FAILURE;
    }

    const char* message = "Hello, World!";
    for (int i = 0; i < N; ++i) {
        if (send_message(hSerial, message) == 0) {
            printf("Message %d sent successfully\n", i + 1);
        }
        else {
            printf("Failed to send message %d\n", i + 1);
        }
        sleep_ms(delay_ms);
    }

    CloseHandle(hSerial);

    ASPEP_start(&aspepOverUartA);
    MCP_Over_UartA.pTransportLayer->fSendPacket(MCP_Over_UartA.pTransportLayer, MCP_Over_UartA.txBuffer, MCP_Over_UartA.txLength, MCTL_SYNC);

    return EXIT_SUCCESS;
}


//int main()
//{
//    std::cout << "Hello World!\n";
//}
