/**
 * @file application.hpp
 * @brief Application class and SDL windowing code
 *
 * @author Eric Butler (edbutler)
 */

#ifndef radiosity_APPLICATION_APPLICATION_HPP_
#define radiosity_APPLICATION_APPLICATION_HPP_

#include <SDL/SDL_events.h>

namespace radiosity {

/**
 * A windowed application.
 */
class Application
{
public:

    static int start_application(
        Application* app,
        int width, int height, double fps,
        const char* title );

    Application();

    virtual ~Application();

    /**
     * Initializes the application.
     * @return true on successful initialization, false otherwise.
     * @remark The opengl context is valid by this point.
     */
    virtual bool initialize() = 0;

    /**
     * Cleans up any resources used by the application.
     */
    virtual void destroy() = 0;

    /**
     * Updates the application for the current frame.
     * Invoked on the update thread.
     * @param delta_time The length of the time step in seconds.
     */
    virtual void update( double delta_time ) = 0;

    /**
     * Renders the application for the current frame.
     * Invoked on the render thread.
     */
    virtual void render() = 0;

    /**
     * Handle an event. Invoked on the update thread.
     */
    virtual void handle_event( const SDL_Event& event ) = 0;

    /**
     * Terminates the application on the next frame.
     */
    void end_main_loop();

    /**
     * Query the current dimension of the window.
     */
    void get_dimension( int* width, int* height ) const;

    /**
     * Takes a screenshot using a default name
     */
    void take_screenshot();

private:

    // in case multiple threads are used
    volatile bool running;

    // no meaningful assignment/copy
    Application( const Application& );
    Application& operator=( const Application& );
};

} /* radiosity */

#endif /* radiosity_APPLICATION_APPLICATION_HPP_ */

