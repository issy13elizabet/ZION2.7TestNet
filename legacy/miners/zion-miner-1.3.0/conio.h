#pragma once

#ifndef CONIO_LINUX_H
#define CONIO_LINUX_H

// Linux replacement for Windows conio.h functions

#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes for Windows conio.h compatibility
int _kbhit(void);
int _getch(void);
void clrscr(void);
void gotoxy(int x, int y);

#ifdef __cplusplus
}
#endif

// Implementation of conio functions for Linux
inline int _kbhit(void) {
    static const int STDIN = 0;
    static bool initialized = false;
    
    if (!initialized) {
        // Use termios to turn off line buffering
        struct termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        tcsetattr(STDIN, TCSANOW, &term);
        setbuf(stdin, NULL);
        initialized = true;
    }
    
    int bytesWaiting;
    ioctl(STDIN, FIONREAD, &bytesWaiting);
    return bytesWaiting;
}

inline int _getch(void) {
    char buf = 0;
    struct termios old = {0};
    
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
        
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    
    if (tcsetattr(0, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
        
    if (read(0, &buf, 1) < 0)
        perror ("read()");
        
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror ("tcsetattr ~ICANON");
        
    return (int)buf;
}

inline void clrscr(void) {
    printf("\033[2J\033[H");
}

inline void gotoxy(int x, int y) {
    printf("\033[%d;%dH", y, x);
}

#endif // CONIO_LINUX_H