// contribs/rlev/src/main/java/org/matsim/contrib/rlev/RewardProbe.java
package org.matsim.contrib.rlev;

import org.matsim.api.core.v01.events.*;
import org.matsim.api.core.v01.events.handler.*;
import org.matsim.core.controler.listener.*;
import org.matsim.core.controler.events.*;
import org.matsim.api.core.v01.Id;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Collects simple metrics for reward:
 * - average leg duration (sec)
 * - crude charging integral proxy (count/weight of charging events, adjust to your EV contrib if available)
 */
public class RewardProbe implements StartupListener, ShutdownListener,
        PersonDepartureEventHandler, PersonArrivalEventHandler {

    private final Map<Id, Double> legStart = new HashMap<>();
    private final AtomicLong legs = new AtomicLong(0);
    private double legDurationSum = 0.0;

    // If you have EV contrib events (ChargingStartEvent/ChargingEndEvent), hook them here.
    // For now, simple proxy via "pt" or "car" dwell time can be added if you have a signal.

    @Override
    public void notifyStartup(StartupEvent event) {
        // nothing
    }

    @Override
    public void notifyShutdown(ShutdownEvent event) {
        // nothing
    }

    @Override
    public void handleEvent(PersonDepartureEvent e) {
        legStart.put(e.getPersonId(), e.getTime());
    }

    @Override
    public void handleEvent(PersonArrivalEvent e) {
        Double s = legStart.remove(e.getPersonId());
        if (s != null) {
            legDurationSum += (e.getTime() - s);
            legs.incrementAndGet();
        }
    }

    public double getAvgLegDurationSec() {
        long n = legs.get();
        return (n > 0) ? (legDurationSum / n) : 0.0;
    }

    // Stub for charge proxy; replace with proper EV events if you have them wired:
    public double getChargeIntegralProxy() {
        // Return 0..1 normalized proxy (you can wire real charging events later)
        return 0.0;
    }
}
